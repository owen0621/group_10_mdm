import os
import math
import numpy as np
import torch
from sample.gpt_api import PromptSplitter
from data_loaders.humanml.utils.my_plot_script import plot_3d_motion
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from utils import dist_util
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.scripts.motion_process import recover_from_ric


class LongMotionGenerator:
    def __init__(self, args):
        self.args = args
        self.device = dist_util.setup_dist(args.device)
        self.args.batch_size = 1               # 一次只生成一個 clip
        self.max_frames = 196                  # 模型最大支援幀數
        self.fps = 12.5 if args.dataset == 'kit' else 20  
        self.clip_seconds = self.max_frames / self.fps  

        print("Loading model…")
        self.data = get_dataset_loader(
            name=args.dataset,
            batch_size=1,
            num_frames=self.max_frames,
            split='test',
            hml_mode='train'
        )
        self.model, self.diffusion = create_model_and_diffusion(args, self.data)
        load_saved_model(self.model, args.model_path, use_avg=args.use_ema)
        self.model = ClassifierFreeSampleModel(self.model).to(self.device)
        self.model.eval()

        # 骨架拓撲 (KIT 或 HumanML)
        self.skeleton = (
            paramUtil.kit_kinematic_chain 
            if args.dataset == 'kit' 
            else paramUtil.t2m_kinematic_chain
        )
        self.prompt_splitter = PromptSplitter(model="gpt-4")

        # delete_frames：先刪尾 20 幀
        self.delete_frames = 20
        # supply_frames：從「刪除後剩餘部分」取最後 20 幀，供下一段 inpainting
        self.supply_frames = 20
        self.compensation_frames = 1  # 每段 clip 加 2 秒來補償刪除的幀數

    def generate_from_prompt(self,
                             text_prompt: str,
                             duration_seconds: int,
                             per_clip_seconds: float,
                             output_path: str = "./long_motion.mp4"):
        """
        :param text_prompt:      長影片的完整文字描述
        :param duration_seconds: 最終要生成的總時長（秒）
        :param per_clip_seconds: （此參數已不使用，改由 GPT 回傳的秒數決定）
        :param output_path:      最後輸出影片檔路徑
        """
        # 1) 先得知總共需要多少段 clip
        #    仍沿用 per_clip_seconds 來估算片段數量，但分段後會用 GPT 指定的秒數
        num_clips = int(math.ceil(duration_seconds / per_clip_seconds))
        segments_with_dur = self.prompt_splitter.split_prompt(text_prompt, num_clips, with_duration=True)
        # segments_with_dur 為 [(text1, dur1), (text2, dur2), …]

        print(f"Splitted into {len(segments_with_dur)} segments (GPT 指定秒數)。")

        # 2) 準備一次迭代 dataset
        iterator = iter(self.data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(self.device)

        all_motions_rep = []    # 儲存最終每段「刪除 40 幀後」的 representation
        prev_overlap_rep = None # 暫存下一段 inpainting 用的 (J, F, supply_frames)

        for i, (seg_text, seg_dur) in enumerate(segments_with_dur):
            # GPT 回傳的 seg_dur 必須介於 3-8 秒，我們要加 2 秒來補償 40 幀的刪除
            clip_seconds_i = seg_dur + self.compensation_frames
            frames_per_clip_i = int(round(clip_seconds_i * self.fps))
            frames_per_clip_i = min(frames_per_clip_i, self.max_frames)

            if frames_per_clip_i <= self.delete_frames + self.supply_frames:
                raise ValueError(
                    f"第 {i+1} 段 clip 的 frames_per_clip ({frames_per_clip_i}) "
                    f"必須大於 delete_frames + supply_frames ({self.delete_frames + self.supply_frames})!"
                )

            print(f"[Clip {i+1}/{len(segments_with_dur)}] \"{seg_text}\" "
                  f"→ GPT 秒數={seg_dur}s, 加{self.compensation_frames}s補償後={clip_seconds_i}s, 幀數={frames_per_clip_i}")

            # 每段 clip 都要更新 model_kwargs['y']['lengths']
            model_kwargs['y']['lengths'] = torch.tensor([frames_per_clip_i]).to(self.device)
            model_kwargs['y']['text'] = [seg_text]
            model_kwargs['y']['scale'] = torch.tensor(
                [self.args.guidance_param], device=self.device
            )

            # 若不是第一段，就先做 inpainting
            if i > 0:
                inpainted = torch.zeros(
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip_i),
                    device=self.device
                )
                # 將上一段剩下部分的最後 supply_frames 幀填到此段開頭
                inpainted[..., 0:self.supply_frames] = prev_overlap_rep
                mask = torch.zeros_like(inpainted, dtype=torch.bool)
                mask[..., 0:self.supply_frames] = True

                model_kwargs['y']['inpainted_motion'] = inpainted
                model_kwargs['y']['inpainting_mask'] = mask

            # 產生這一段 clip 的 representation
            sample_rep = self.diffusion.p_sample_loop(
                self.model,
                (1, self.model.njoints, self.model.nfeats, frames_per_clip_i),
                model_kwargs=model_kwargs,
                clip_denoised=False,
                skip_timesteps=0,
                progress=True,
            )

            # 3) 移到 CPU 並轉 numpy
            sample_rep = sample_rep.cpu()  # shape: (1, J, F, frames_per_clip_i)
            clip_rep_full = sample_rep.permute(0, 3, 1, 2).squeeze(0).numpy()  
            # clip_rep_full shape = (frames_per_clip_i, J, F)

            # 4) 先刪掉尾巴 delete_frames 幀
            truncated_after_delete = clip_rep_full[:-self.delete_frames]
            # 這時 truncated_after_delete.shape = (frames_per_clip_i - delete_frames, J, F)

            # 5) 從 truncated_after_delete 取倒數 supply_frames 幀，供下一段 inpainting
            overlap = truncated_after_delete[-self.supply_frames:]  # shape = (supply_frames, J, F)
            prev_overlap_rep = np.transpose(overlap, (1, 2, 0))      # shape = (J, F, supply_frames)
            prev_overlap_rep = torch.from_numpy(prev_overlap_rep).to(self.device)

            # 6) 再把 truncated_after_delete 的最後 supply_frames 幀刪掉，
            #    剩下 (frames_per_clip_i - delete_frames - supply_frames) 幀，存入 all_motions_rep
            truncated_to_store = truncated_after_delete[:-self.supply_frames]  
            all_motions_rep.append(truncated_to_store)

        # 7) 合併所有段落（它們都已刪掉 40 幀 = 20+20）
        full_rep = np.concatenate(all_motions_rep, axis=0)  # shape = (T_total, J, F)

        # 8) 根據 data_rep 還原到 XYZ
        if self.model.data_rep == 'hml_vec':
            T_total = full_rep.shape[0]
            rep_tensor = torch.from_numpy(full_rep).reshape(1, T_total, -1).float()
            rep_xyz = self.data.dataset.t2m_dataset.inv_transform(rep_tensor).float()
            n_joints = 22 if rep_xyz.shape[2] // 3 == 263 // 3 else 21
            rep_xyz = recover_from_ric(rep_xyz, n_joints).numpy()[0]  # (T_total, J, 3)
        else:
            rep_xyz = full_rep  # 如果 data_rep 已經是 'xyz'，full_rep 本身就是 (T_total, J, 3)

        # 9) 最後一次呼叫 plot_3d_motion 寫出完整影片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"[LongMotionGenerator] Writing final video ({rep_xyz.shape[0]} frames) to: {output_path}")
        plot_3d_motion(
            save_path=output_path,
            kinematic_tree=self.skeleton,
            joints=rep_xyz,
            title=text_prompt,
            dataset=self.args.dataset,
            fps=self.fps
        )
        print(f"[LongMotionGenerator] Done. Saved: {output_path}")


# Example usage
if __name__ == "__main__":
    from utils.parser_util import edit_args

    args = edit_args()
    args.text_condition = ""
    args.guidance_param = 2.5
    args.edit_mode = 'first_frame_only'
    #TODO
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    gen = LongMotionGenerator(args)
    prompt = (
        # "A person runs forward with quick steps, transitions into a spinning turn to the left, "
        # "immediately jumps into the air with arms extended, and lands smoothly before continuing "
        # "with a side step to the right."
        
        #demo2
        # "Beginning with a slow walk, the person accelerates into a sprint, leaps over an imaginary obstacle with arms raised, lands in a low squat for a while, and then rises smoothly into a balanced pose and turn backward walking."
        
        #demo3
        "A person swiftly jogs forward, then pivots sharply to the right, executing a graceful spin with arms extended, followed by a high leap into the air, landing softly and transitioning into a backward step with a slight bow."
    )

    # 這裡傳入的 per_clip_seconds 只是用來估算段數，實際每段秒數由 GPT 回傳決定
    gen.generate_from_prompt(
        text_prompt=prompt,
        duration_seconds=30,
        per_clip_seconds=6.0,  # 用來計算 clip 數，實際秒數由 GPT 回傳
        #TODO
        # output_path="./outputs/long_dance_clip_modified_dynamic.mp4"
        output_path="./outputs/new_appr_demo3.mp4"
    )
