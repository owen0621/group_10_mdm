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

        # delete_frames：先刪尾 10 幀
        self.delete_frames = 20
        # supply_frames：從「刪除後剩餘部分」取最後 20 幀，供下一段 inpainting
        self.supply_frames = 20

    def generate_from_prompt(self,
                             text_prompt: str,
                             duration_seconds: int,
                             per_clip_seconds: float,
                             output_path: str = "./long_motion.mp4"):
        """
        :param text_prompt:      長影片的完整文字描述
        :param duration_seconds: 最終要生成的總時長（秒）
        :param per_clip_seconds: 每個小 clip 的時長（秒），必須 > (delete_frames + supply_frames) / fps
        :param output_path:      最後輸出影片檔路徑
        """
        # 1) 根據 per_clip_seconds 算出 frames_per_clip
        frames_per_clip = int(round(per_clip_seconds * self.fps))
        frames_per_clip = min(frames_per_clip, self.max_frames)
        if frames_per_clip <= self.delete_frames + self.supply_frames:
            raise ValueError(
                f"frames_per_clip ({frames_per_clip}) 必須大於 delete_frames + supply_frames ({self.delete_frames + self.supply_frames})!"
            )

        # 2) 計算需要多少段 clip
        num_clips = int(math.ceil(duration_seconds / per_clip_seconds))
        sub_prompts = self.prompt_splitter.split_prompt(text_prompt, num_clips, with_duration=False)

        print(f"Generating {num_clips} clips for {duration_seconds}s "
              f"({frames_per_clip} frames per clip; delete {self.delete_frames}, supply {self.supply_frames})…")

        # 3) 取得 dataset iterator，並把每段 clip 的 lengths 設為 frames_per_clip
        iterator = iter(self.data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(self.device)
        model_kwargs['y']['lengths'] = torch.tensor([frames_per_clip])

        all_motions_rep = []    # 儲存每段「刪除 30 幀後」的 representation (frames_per_clip - 30, J, F)
        prev_overlap_rep = None # 暫存上一段「刪除 10 後倒數 20 幀」 → (J, F, supply_frames)

        for i in range(num_clips):
            print(f"[Clip {i+1}/{num_clips}] \"{sub_prompts[i]}\"")
            model_kwargs['y']['text'] = [sub_prompts[i]]
            model_kwargs['y']['scale'] = torch.tensor(
                [self.args.guidance_param], device=self.device
            )

            if i == 0:
                # 第一段 clip：直接生成完整 frames_per_clip 幀
                sample_rep = self.diffusion.p_sample_loop(
                    self.model,
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip),
                    model_kwargs=model_kwargs,
                    clip_denoised=False,
                    skip_timesteps=0,
                    progress=True,
                )
            else:
                # 後續段：inpaint 前 supply_frames 幀
                inpainted = torch.zeros(
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip),
                    device=self.device
                )
                # prev_overlap_rep shape = (J, F, supply_frames)
                inpainted[..., 0:self.supply_frames] = prev_overlap_rep

                mask = torch.zeros_like(inpainted, dtype=torch.bool)
                mask[..., 0:self.supply_frames] = True

                model_kwargs['y']['inpainted_motion'] = inpainted
                model_kwargs['y']['inpainting_mask'] = mask

                sample_rep = self.diffusion.p_sample_loop(
                    self.model,
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip),
                    model_kwargs=model_kwargs,
                    clip_denoised=False,
                    skip_timesteps=0,
                    progress=True,
                )

            # 4) 移到 CPU 並轉 numpy
            sample_rep = sample_rep.cpu()  # shape: (1, J, F, frames_per_clip)
            clip_rep_full = sample_rep.permute(0, 3, 1, 2).squeeze(0).numpy()  # (frames_per_clip, J, F)

            # 5) 從「刪掉 10 幀後剩下部分」取倒數 20 幀，供下一段 inpaint
            truncated_after_delete = clip_rep_full[:-self.delete_frames]  # shape = (frames_per_clip-delete, J, F)
            # 取倒數 20 幀： indices [ -supply_frames : ]
            overlap = truncated_after_delete[-self.supply_frames:]  # shape = (supply_frames, J, F)
            # 變成 (J, F, supply_frames) 方便下一 loop 塞入 inpainted
            prev_overlap_rep = np.transpose(overlap, (1, 2, 0))
            prev_overlap_rep = torch.from_numpy(prev_overlap_rep).to(self.device)

            # 6) 再把 trunc_after_delete 刪掉最後那 20 幀 → truncated_to_store
            truncated_to_store = truncated_after_delete[:-self.supply_frames]  
            # shape = (frames_per_clip - delete_frames - supply_frames, J, F)

            # 7) 把這段「已刪 10 + 刪 20 重疊」的部分存到 all_motions_rep
            all_motions_rep.append(truncated_to_store)

        # 8) 合併所有段落（它們都已經被刪掉 30 幀）
        full_rep = np.concatenate(all_motions_rep, axis=0)  # shape = (T_total, J, F)

        # 9) 根據 data_rep 還原到 XYZ
        if self.model.data_rep == 'hml_vec':
            T_total = full_rep.shape[0]
            rep_tensor = torch.from_numpy(full_rep).reshape(1, T_total, -1).float()
            rep_xyz = self.data.dataset.t2m_dataset.inv_transform(rep_tensor).float()
            n_joints = 22 if rep_xyz.shape[2] // 3 == 263 // 3 else 21
            rep_xyz = recover_from_ric(rep_xyz, n_joints).numpy()[0]  # (T_total, J, 3)
        else:
            # 如果 data_rep 已經是 'xyz'，full_rep 本身就是 (T_total, J, 3)
            rep_xyz = full_rep

        # 10) 一次呼叫 plot_3d_motion 寫出長影片
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

    gen = LongMotionGenerator(args)
    prompt = (
        "A person runs forward with quick steps, transitions into a spinning turn to the left, "
        "immediately jumps into the air with arms extended, and lands smoothly before continuing "
        "with a side step to the right."
    )

    # 例：每段 6 秒 → frames_per_clip = 6 * fps，總長 30 秒 → 5 段 clip
    gen.generate_from_prompt(
        text_prompt=prompt,
        duration_seconds=30,
        per_clip_seconds=6.0,
        output_path="./outputs/long_dance_clip_modified_fix.mp4"
    )
