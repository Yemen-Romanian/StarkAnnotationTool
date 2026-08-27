import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import time

import cv2 as cv
from pathlib import Path

from lib.test.evaluation import Tracker
from lib.utils.device import get_device, set_device


# Bounding box written for skipped frames (object absent / occluded).
SKIP_BOX = [0, 0, 0, 0]


def build_tracker(tracker_name, tracker_param):
    """Create a ready-to-use tracker instance (params + network)."""
    print("Running the tracker on device: %s" % get_device())
    tracker = Tracker(tracker_name, tracker_param, "video")
    params = tracker.get_parameters()
    params.tracker_name = tracker_name
    params.param_name = tracker_param
    params.debug = 0
    return tracker.create_tracker(params)


def select_box(window_name, frame_bgr, prompt):
    """Ask the user to draw a box with the mouse. Returns [x, y, w, h]."""
    frame_disp = frame_bgr.copy()
    cv.putText(frame_disp, prompt, (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
               1.2, (0, 0, 255), 2)
    x, y, w, h = cv.selectROI(window_name, frame_disp, fromCenter=False)
    return [int(x), int(y), int(w), int(h)]


def draw_overlay(frame_bgr, box, frame_idx, total_frames, timing=None):
    """Draw the current box and the control hints on a copy of the frame."""
    frame_disp = frame_bgr.copy()
    if box is not None and box != SKIP_BOX:
        x, y, w, h = (int(round(v)) for v in box)
        cv.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 3)

    if total_frames > 0:
        frame_text = "Frame %d / %d" % (frame_idx, total_frames - 1)
    else:
        frame_text = "Frame %d" % frame_idx
    lines = [
        frame_text,
        "SPACE: accept box    s: skip (0,0,0,0)",
        "c: correct & re-init    ENTER: save",
        "q: quit & save",
    ]
    if timing is not None:
        lines.append(timing)
    for i, text in enumerate(lines):
        y0 = 30 + i * 28
        cv.putText(frame_disp, text, (20, y0), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.0, (0, 0, 0), 3)
        cv.putText(frame_disp, text, (20, y0), cv.FONT_HERSHEY_COMPLEX_SMALL,
                   1.0, (0, 255, 255), 1)
    return frame_disp


def save_annotations(boxes, output_path):
    """Write one comma-separated 'x,y,w,h' row per frame (overwrites the file)."""
    with open(output_path, "w") as f:
        for box in boxes:
            f.write("%.4f,%.4f,%.4f,%.4f\n" % (box[0], box[1], box[2], box[3]))
    print("Saved %d annotations to %s" % (len(boxes), output_path))


def load_annotations(output_path):
    """Load previously saved 'x,y,w,h' rows. Returns [] if the file is absent."""
    boxes = []
    if not os.path.isfile(output_path):
        return boxes
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                continue
            boxes.append([float(v) for v in parts[:4]])
    return boxes


def run_annotation(tracker_name, tracker_param, videofile):
    assert os.path.isfile(videofile), "Invalid video path: {}".format(videofile)

    tracker = build_tracker(tracker_name, tracker_param)

    video_path = Path(videofile)
    output_path = str(video_path.with_name(video_path.stem + "_groundtruth.txt"))

    cap = cv.VideoCapture(videofile)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    success, frame = cap.read()
    if not success:
        print("Read frame from {} failed.".format(videofile))
        cap.release()
        return

    window_name = "Annotation: " + tracker_name
    cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    boxes = load_annotations(output_path)

    if boxes:
        # ---- Resume: replay the already-recorded frames so the video position
        # lands right after the last record, initializing the tracker on the
        # most recent valid (non-skip) box along the way. ----
        print("Found %d existing annotations, resuming after frame %d." % (len(boxes), len(boxes) - 1))
        last_valid_frame = None
        last_valid_box = None
        # frame 0 was already read above; the loop reads frames 1..len(boxes)-1.
        for idx in range(len(boxes)):
            if idx > 0:
                success, frame = cap.read()
                if not success or frame is None:
                    print("Video has fewer frames than the saved annotation. Nothing left to do.")
                    cap.release()
                    cv.destroyAllWindows()
                    return
            if boxes[idx] != SKIP_BOX:
                last_valid_frame = frame
                last_valid_box = boxes[idx]
        frame_idx = len(boxes) - 1
        if last_valid_box is None:
            print("Existing annotations contain no valid box to resume tracking from. Aborting.")
            cap.release()
            cv.destroyAllWindows()
            return
        tracker.initialize(cv.cvtColor(last_valid_frame, cv.COLOR_BGR2RGB), {'init_bbox': last_valid_box})
    else:
        # ---- Frame 0: manual initialization (first row of ground-truth) ----
        init_box = select_box(window_name, frame, "Select target and press ENTER (initial frame)")
        if init_box == SKIP_BOX:
            print("No initial box selected. Aborting.")
            cap.release()
            cv.destroyAllWindows()
            return
        tracker.initialize(cv.cvtColor(frame, cv.COLOR_BGR2RGB), {'init_bbox': init_box})
        boxes.append(init_box)
        frame_idx = 0

    # ---- Subsequent frames: track, then wait for the user's decision ----
    total_track_time = 0.0
    tracked_frames = 0
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        frame_idx += 1

        t0 = time.time()
        out = tracker.track(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        track_time = time.time() - t0
        pred_box = [float(s) for s in out['target_bbox']]

        total_track_time += track_time
        tracked_frames += 1
        timing_text = "track: %.0f ms (%.1f FPS)   avg: %.0f ms (%.1f FPS)" % (
            track_time * 1000, 1.0 / max(track_time, 1e-6),
            total_track_time / tracked_frames * 1000, tracked_frames / total_track_time)

        # Wait for a decision for this frame.
        decision = None
        while decision is None:
            cv.imshow(window_name, draw_overlay(frame, pred_box, frame_idx, total_frames, timing_text))
            key = cv.waitKey(20) & 0xFF

            if key == ord(' '):          # accept tracker prediction
                boxes.append(pred_box)
                decision = 'next'
            elif key == ord('s'):        # skip this frame
                boxes.append(list(SKIP_BOX))
                decision = 'next'
            elif key == ord('c'):        # correct manually and re-initialize
                corrected = select_box(window_name, frame,
                                       "Correction: select target and press ENTER")
                if corrected == SKIP_BOX:
                    continue  # cancelled selection, keep asking for this frame
                tracker.initialize(cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                                   {'init_bbox': corrected})
                boxes.append(corrected)
                pred_box = corrected
                decision = 'next'
            elif key in (13, 10):        # ENTER: write current results (overwrite)
                save_annotations(boxes, output_path)
            elif key == ord('q'):        # quit and save what we have
                decision = 'quit'

        if decision == 'quit':
            break

    cap.release()
    cv.destroyAllWindows()

    if tracked_frames:
        print("Tracked %d frames on %s: %.1f ms/frame (%.1f FPS average)" % (
            tracked_frames, get_device(), total_track_time / tracked_frames * 1000,
            tracked_frames / total_track_time))

    save_annotations(boxes, output_path)


def main():
    parser = argparse.ArgumentParser(description='Annotate a video with bounding boxes using the STARK tracker.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run the tracker on: 'cpu', 'cuda', 'cuda:1', ... "
                             "Defaults to CUDA when available, otherwise CPU.")

    args = parser.parse_args()
    if args.device is not None:
        set_device(args.device)
    run_annotation(args.tracker_name, args.tracker_param, args.videofile)


if __name__ == '__main__':
    main()
