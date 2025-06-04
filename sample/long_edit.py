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
        # 預設每段 clip 長度（秒）對應的幀數（但實際會再被 per_clip_seconds 覆蓋）
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
        self.skeleton = (paramUtil.kit_kinematic_chain 
                         if args.dataset == 'kit' 
                         else paramUtil.t2m_kinematic_chain)
        self.prompt_splitter = PromptSplitter(model="gpt-4")

        # overlap_frames：每段 clip 與下一段之間要「銜接」的幀數
        self.overlap_frames = 10

    def generate_from_prompt(self,
                             text_prompt: str,
                             duration_seconds: int,
                             per_clip_seconds: float,
                             output_path: str = "./long_motion.mp4"):
        """
        :param text_prompt:      長影片的整段文字條件
        :param duration_seconds: 最終要生成的總時長（秒）
        :param per_clip_seconds: 每個小 clip 的時長（秒），必須 ≤ max_frames/fps
        :param output_path:      最後輸出影片檔路徑
        """
        # 1) 根據 per_clip_seconds 算出用多少 frames
        frames_per_clip = int(round(per_clip_seconds * self.fps))
        # 別超過模型最大支援幀數
        frames_per_clip = min(frames_per_clip, self.max_frames)
        if frames_per_clip <= self.overlap_frames:
            raise ValueError(
                f"per_clip_seconds ({per_clip_seconds}s) * fps ({self.fps}) "
                f"must exceed overlap_frames ({self.overlap_frames})!"
            )

        # 2) 計算需要多少個 clip
        num_clips = int(math.ceil(duration_seconds / per_clip_seconds))
        sub_prompts = self.prompt_splitter.split_prompt(text_prompt, num_clips)

        print(f"Generating {num_clips} clips for {duration_seconds}s "
              f"({frames_per_clip} frames per clip)…")

        # 3) 取得 dataset iterator，並記得把每段 clip 的 lengths 設成 frames_per_clip
        iterator = iter(self.data)
        input_motion, model_kwargs = next(iterator)
        input_motion = input_motion.to(self.device)
        model_kwargs['y']['lengths'] = torch.tensor([frames_per_clip])

        all_motions_rep = []     # 存放每段 clip 在 model representation 上 (frames_per_clip, J, F)
        prev_last_rep_overlap = None  # 暫存「上一段最後 overlap_frames 幀的 representation」

        for i in range(num_clips):
            print(f"[Clip {i+1}/{num_clips}] \"{sub_prompts[i]}\"")
            # 設定文字條件 & CFG scale
            model_kwargs['y']['text'] = [sub_prompts[i]]
            model_kwargs['y']['scale'] = torch.tensor(
                [self.args.guidance_param], device=self.device
            )

            if i == 0:
                # 第一段 clip：全新生成 frames_per_clip 幀
                sample_rep = self.diffusion.p_sample_loop(
                    self.model,
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip),
                    model_kwargs=model_kwargs,
                    clip_denoised=False,
                    skip_timesteps=0,
                    progress=True,
                )
            else:
                # 後續段 clip：用上一段最後 overlap_frames 幀 inpainting
                inpainted = torch.zeros(
                    (1, self.model.njoints, self.model.nfeats, frames_per_clip),
                    device=self.device
                )
                # 將上一段最後 overlap_frames 幀放到這一段的前面
                # prev_last_rep_overlap shape = (J, F, overlap_frames)
                inpainted[..., 0:self.overlap_frames] = prev_last_rep_overlap

                mask = torch.zeros_like(inpainted, dtype=torch.bool)
                mask[..., 0:self.overlap_frames] = True

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

            # 4) 將 sample_rep 移到 CPU 並擷取最後 overlap_frames 幀供下一段使用
            sample_rep = sample_rep.cpu()  # shape: (1, J, F, frames_per_clip)
            prev_last_rep_overlap = sample_rep[0][..., -self.overlap_frames:].clone()

            # 5) 把 sample_rep 轉成 numpy (frames_per_clip, J, F)，放到 all_motions_rep
            clip_rep = sample_rep.permute(0, 3, 1, 2).squeeze(0).numpy()  # shape: (frames_per_clip, J, F)
            all_motions_rep.append(clip_rep)

        # 6) 所有 clip 生成完後，在 representation 空間做「刪掉前一段 clip 的後 overlap_frames 幀」再拼接
        merged_rep = []
        for idx in range(num_clips):
            if idx < num_clips - 1:
                # 如果不是最後一段，就取這一段除去「最後 overlap_frames 幀」(避免下一段重複)
                merged_rep.append(all_motions_rep[idx][:-self.overlap_frames])
            else:
                # 最後一段保留完整
                merged_rep.append(all_motions_rep[idx])

        # 合併所有段落
        full_rep = np.concatenate(merged_rep, axis=0)  # shape = (T_total, J, F)

        # 7) 根據 data_rep 還原成 XYZ
        if self.model.data_rep == 'hml_vec':
            T_total = full_rep.shape[0]
            # full_rep → tensor 形狀 (1, T_total, J*F)
            rep_tensor = torch.from_numpy(full_rep).reshape(1, T_total, -1).float()
            # inv_transform → (1, T_total, J*3)，再 recover → (1, T_total, J, 3)
            rep_xyz = self.data.dataset.t2m_dataset.inv_transform(rep_tensor).float()
            n_joints = 22 if rep_xyz.shape[2] // 3 == 263 // 3 else 21
            rep_xyz = recover_from_ric(rep_xyz, n_joints).numpy()[0]  # shape: (T_total, J, 3)
        else:
            # 如果 data_rep 已經是 'xyz'，full_rep 本身就是 (T_total, J, 3)
            rep_xyz = full_rep

        # 8) 最後一次呼叫 plot_3d_motion 寫出長影片
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
        "A person stands still for a moment, takes a few quick steps backward, suddenly crouches down, and stands up again, rolls sideways to the left while extending their right arm forward."
    )

    # 範例：想用每段 5 秒（= 5 * fps 幀），總長 20 秒 → 20 ÷ 5 = 4 段 clip
    gen.generate_from_prompt(
        text_prompt=prompt,
        duration_seconds=30,
        per_clip_seconds=6,
        output_path="./outputs/long_dance_clip.mp4"
    )
