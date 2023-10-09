import os
import subprocess
import sys

import cv2
import torch
import numpy as np
import gradio as gr

from util import resolve_relative_path
from SegTracker import SegTracker
from model_args import segtracker_args, sam_args, aot_args
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video


def clean():
    return None, None, None, None, None, None, [[], []]


def get_click_prompt(click_stack, point):

    click_stack[0].append(point["coord"])
    click_stack[1].append(point["mode"])

    prompt = {
        "points_coord": click_stack[0],
        "points_mode": click_stack[1],
        "multimask": "True",
    }

    return prompt


def get_meta_from_video(input_video):
    if input_video is None:
        return None, None, None, ""

    print("获取输入视频的元信息")
    cap = cv2.VideoCapture(input_video)

    _, first_frame = cap.read()
    cap.release()

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    return first_frame, first_frame, first_frame, ""


def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker


def init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):

    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""


def init_SegTracker_Stroke(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side,
                           origin_frame):

    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame

    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term

    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame


def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term,
                                    sam_gap, max_obj_num, points_per_side):

    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]

    print("撤销！")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][:-1]
        click_stack[1] = click_stack[1][:-1]

    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord": click_stack[0],
            "points_mode": click_stack[1],
            "multimask": "True",
        }

        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
        origin_frame=origin_frame,
        coords=np.array(prompt["points_coord"]),
        modes=np.array(prompt["points_mode"]),
        multimask=prompt["multimask"],
    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    return masked_frame


def sam_click(Seg_Tracker, origin_frame, point_mode, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap,
              max_obj_num, points_per_side, evt: gr.SelectData):
    """
    Args:
        origin_frame: nd.array
        click_stack: [[coordinate], [point_mode]]
    """

    print("Click")

    if point_mode == "Positive":
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        point = {"coord": [evt.index[0], evt.index[1]], "mode": 0}

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
                                               points_per_side, origin_frame)

    # get click prompts for sam to predict mask
    click_prompt = get_click_prompt(click_stack, point)

    # Refine acc to prompt
    masked_frame = seg_acc_click(Seg_Tracker, click_prompt, origin_frame)

    return Seg_Tracker, masked_frame, click_stack


def add_new_object(Seg_Tracker):

    prev_mask = Seg_Tracker.first_frame_mask
    Seg_Tracker.update_origin_merged_mask(prev_mask)
    Seg_Tracker.curr_idx += 1

    print("开始准备添加新对象！")

    return Seg_Tracker, [[], []]


def tracking_objects(Seg_Tracker, input_video, input_img_seq=None, frame_num=0):
    print("开始追踪")
    return tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, frame_num)


def remove_watermark(input_video):
    print("开始去除水印")
    print('cwd', os.getcwd())
    os.chdir('./ProPainter')
    print('cwd', os.getcwd())
    inference = resolve_relative_path('./ProPainter/inference_propainter.py')

    video_name = os.path.basename(input_video).split('.')[0].split('_')[0]
    output_base_path = resolve_relative_path('./output/')
    output_path = f'{output_base_path}/{video_name}/'
    mask = f'{output_path}/{video_name}_masks/'

    command = f'python {inference} --video {input_video} --mask {mask}  --output {output_path} --fp16 --subvideo_length 50'
    print(command)
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        error_message = result.stderr.decode('utf-8', 'ignore')
        print(f"错误 {error_message}")
    else:
        print("成功")
    file_name = input_video.split('\\')[-1].split('.')[0]
    print(file_name)
    os.chdir(resolve_relative_path('./'))
    print('cwd', os.getcwd())
    return output_path + '/' + file_name + '/' + 'inpaint_out' + '.mp4'
    # return input_video


def seg_track_app():

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks()

    with app:
        gr.Markdown('''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">视频去水印</span>
            </div>
            ''')
        gr.Markdown('## 第一步：生成蒙版')
        click_stack = gr.State([[], []])  # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row():
            # video input
            input_video = gr.Video(label='待处理视频', height=400)

            input_first_frame = gr.Image(label='选择蒙版对象', interactive=True, height=400)

        with gr.Row():
            with gr.Column():
                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_mode = gr.Radio(choices=["Positive", "Negative"],
                                              value="Positive",
                                              label="Point Prompt",
                                              interactive=True)

                        # args for modify and tracking
                        click_undo_but = gr.Button(value="撤销", interactive=True)

            with gr.Column():
                with gr.Tab(label="SegTracker Args", visible=False):
                    # args for tracking in video do segment-everthing
                    points_per_side = gr.Slider(label="points_per_side",
                                                minimum=1,
                                                step=1,
                                                maximum=100,
                                                value=16,
                                                interactive=True,
                                                visible=False)

                    sam_gap = gr.Slider(label='sam_gap',
                                        minimum=1,
                                        step=1,
                                        maximum=9999,
                                        value=100,
                                        interactive=True,
                                        visible=False)

                    max_obj_num = gr.Slider(label='max_obj_num',
                                            minimum=50,
                                            step=1,
                                            maximum=300,
                                            value=255,
                                            interactive=True,
                                            visible=False)
                    with gr.Accordion("aot advanced options", open=False, visible=False):
                        aot_model = gr.Dropdown(label="aot_model",
                                                choices=["deaotb", "deaotl", "r50_deaotl"],
                                                value="r50_deaotl",
                                                interactive=True,
                                                visible=False)
                        long_term_mem = gr.Slider(label="long term memory gap",
                                                  minimum=1,
                                                  maximum=9999,
                                                  value=9999,
                                                  step=1,
                                                  visible=False)
                        max_len_long_term = gr.Slider(label="max len of long term memory",
                                                      minimum=1,
                                                      maximum=9999,
                                                      value=9999,
                                                      step=1,
                                                      visible=False)

        with gr.Row():
            new_object_button = gr.Button(value="添加新对象", interactive=True)
            reset_button = gr.Button(
                value="重置",
                interactive=True,
            )
            track_for_video = gr.Button(
                value="开始追踪",
                interactive=True,
            )

        output_video = gr.Video(label='Output video', height=400)

        gr.Markdown('## 第二步：去除水印')
        start_remove_watermark = gr.Button(value="去水印", interactive=True)
        final_video = gr.Video(label='去除后视频', height=400)

        ##########################################################
        ######################  back-end #########################
        ##########################################################

        # listen to the input_video to get the first frame of video
        input_video.change(fn=get_meta_from_video, inputs=[input_video], outputs=[input_first_frame, origin_frame])

        # ------------------- Interactive component -----------------

        # Interactively modify the mask acc click
        input_first_frame.select(fn=sam_click,
                                 inputs=[
                                     Seg_Tracker,
                                     origin_frame,
                                     point_mode,
                                     click_stack,
                                     aot_model,
                                     long_term_mem,
                                     max_len_long_term,
                                     sam_gap,
                                     max_obj_num,
                                     points_per_side,
                                 ],
                                 outputs=[Seg_Tracker, input_first_frame, click_stack])

        # Add new object
        new_object_button.click(fn=add_new_object, inputs=[Seg_Tracker], outputs=[Seg_Tracker, click_stack])

        # Track object in video
        track_for_video.click(fn=tracking_objects, inputs=[
            Seg_Tracker,
            input_video,
        ], outputs=[output_video])

        # Remove watermark in video
        start_remove_watermark.click(fn=remove_watermark, inputs=[output_video], outputs=[final_video])

        # ----------------- Reset and Undo ---------------------------

        # Reset
        reset_button.click(
            fn=init_SegTracker,
            inputs=[aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame],
            outputs=[Seg_Tracker, input_first_frame, click_stack],
            queue=False,
            show_progress=False)

        # Undo click
        click_undo_but.click(fn=undo_click_stack_and_refine_seg,
                             inputs=[
                                 Seg_Tracker,
                                 origin_frame,
                                 click_stack,
                                 aot_model,
                                 long_term_mem,
                                 max_len_long_term,
                                 sam_gap,
                                 max_obj_num,
                                 points_per_side,
                             ],
                             outputs=[Seg_Tracker, input_first_frame, click_stack])

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)


if __name__ == "__main__":
    ffmpeg_exe = resolve_relative_path('env/Library/bin/')
    os.environ['PATH'] = ffmpeg_exe + ';' + os.environ['PATH']
    seg_track_app()