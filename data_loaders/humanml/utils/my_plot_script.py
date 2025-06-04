import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


def plot_3d_motion(save_path,
                   kinematic_tree,
                   joints,
                   title,
                   dataset,
                   figsize=(3, 4),
                   fps=120,
                   radius=3,
                   vis_mode='default',
                   gt_frames=[],
                   return_clip=False):
    """
    :param save_path: 寫檔路徑，或設定 None 且 return_clip=True 時只回傳 VideoClip
    :param kinematic_tree: list of chains, e.g. paramUtil.t2m_kinematic_chain
    :param joints: np.ndarray, shape=(T, J, 3) or (T, J*3)，這裡我們用 (T, J, 3)
    :param title: str 或 list of str（可對每一幀改 title）
    :param dataset: 'kit','humanml', ...
    :param return_clip: True→回傳 VideoClip；False→直接寫出影片到 save_path
    """

    matplotlib.use('Agg')
    title_per_frame = isinstance(title, list)
    if title_per_frame:
        assert len(title) == len(joints), "Title list 要和 frames 一致"
        title = ['\n'.join(wrap(s, 50)) for s in title]
    else:
        title = '\n'.join(wrap(title, 50))

    # 將 joints 轉成 (T, J, 3)
    data = np.array(joints)
    if data.ndim == 2:
        data = data.reshape(len(joints), -1, 3)
    else:
        data = data.copy()

    # apply dataset-specific scaling / flipping
    if dataset == 'kit':
        data *= 0.003
    elif dataset == 'humanml':
        data *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)

    n_frames = data.shape[0]
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    # 把 pelvis 放到地面（y=0）並紀錄 pelvis 在 x,z 平面上的軌跡
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
    colors = colors_orange

    if vis_mode == 'upper_body':
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    # 定義 update function
    def update(t):
        idx = min(n_frames - 1, int(t * fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        if title_per_frame:
            _title = title[idx]
        else:
            _title = title
        _title += f' [{idx}]'
        fig.suptitle(_title, fontsize=8)

        # 畫地板
        xz_min = MINS[0] - trajec[idx, 0]
        xz_max = MAXS[0] - trajec[idx, 0]
        z_min = MINS[2] - trajec[idx, 1]
        z_max = MAXS[2] - trajec[idx, 1]
        verts = [
            [xz_min, 0, z_min],
            [xz_min, 0, z_max],
            [xz_max, 0, z_max],
            [xz_max, 0, z_min]
        ]
        plane = Poly3DCollection([verts])
        plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(plane)

        used_colors = colors_blue if idx in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            lw = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[idx, chain, 0],
                data[idx, chain, 1],
                data[idx, chain, 2],
                linewidth=lw,
                color=color
            )

        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        return mplfig_to_npimage(fig)

    # 用 moviepy 產生 VideoClip，務必指定 duration
    ani = VideoClip(update, duration=n_frames / fps)
    plt.close(fig)

    # 如果回傳 clip，就直接回傳
    if return_clip:
        return ani

    # 否則就寫到檔案
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.write_videofile(
            save_path,
            fps=fps,
            codec="libx264",
            audio=False,
            threads=4,
            logger=None
        )
