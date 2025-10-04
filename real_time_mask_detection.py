#!/usr/bin/env python3
"""
real_time_mask_detection.py
Social distancing + optional mask detection (ONNX) with combined violation logic.

Usage examples:
  # social-distancing only (webcam)
  python3 real_time_mask_detection.py --camera 0

  # video + load saved calibration + birdseye
  python3 real_time_mask_detection.py --video test.mp4 --load-calib --show-birdeye

  # enable face detection and ONNX mask classifier:
  python3 real_time_mask_detection.py --video test.mp4 --load-calib --face-detect --mask-onnx ./classifier/model/mobilenetV2_224.onnx --show-birdeye
"""

import os
import sys
import json
import time
import math
import urllib.request
import argparse
from pathlib import Path

import cv2
import numpy as np

# optional ONNX runtime
try:
    import onnxruntime as ort
except Exception:
    ort = None

# --- Defaults (change URLs if you have local copies) ---
MOBILENET_PROTO = "MobileNetSSD_deploy.prototxt"
MOBILENET_MODEL = "MobileNetSSD_deploy.caffemodel"
MOBILENET_PROTO_URL = "https://gist.githubusercontent.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f/raw/MobileNetSSD_deploy.prototxt"
MOBILENET_MODEL_URL = "https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.caffemodel/download"

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
# alternative mirrors for face model exist; if automatic download 404s, place the file manually.
FACE_MODEL_URLS = [
    "https://huggingface.co/Durraiya/res10_300x300_ssd_iter_140000_fp16.caffemodel/resolve/main/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
]

# MobileNet-SSD VOC classes
CLASSES = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant","sheep",
           "sofa","train","tvmonitor"]


# -------------------------
# Helpers
# -------------------------
def download(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dst} ...")
    urllib.request.urlretrieve(url, dst)
    print("Saved:", dst, "(", os.path.getsize(dst), "bytes )")


def ensure_file(path, url=None, alt_urls=None):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if url:
        try:
            download(url, path)
            return path
        except Exception as e:
            print("Download failed for", url, ":", e)
    if alt_urls:
        for u in alt_urls:
            try:
                download(u, path)
                return path
            except Exception as e:
                print("Alternate download failed for", u, ":", e)
    raise FileNotFoundError(f"{path} not found and could not download.")


# -------------------------
# ONNX mask classifier (robust input-shape handling)
# -------------------------
class ONNXMaskClassifier:
    def __init__(self, model_path):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g. [1,3,224,224] or [1,224,224,3]
        self.output_name = self.sess.get_outputs()[0].name
        print("ONNX model loaded. input_shape:", self.input_shape, "input_name:", self.input_name)

        # canonicalize: determine order and H,W,C
        # shapes often: [N, C, H, W] (NCHW) or [N, H, W, C] (NHWC)
        s = [d for d in self.input_shape]
        # Replace None with -1 for checks
        s_f = [(-1 if d is None else d) for d in s]
        self.is_nchw = False
        if len(s_f) == 4:
            if s_f[1] == 3:
                self.is_nchw = True
                self.C, self.H, self.W = s_f[1], s_f[2], s_f[3]
            elif s_f[3] == 3:
                self.is_nchw = False
                self.H, self.W, self.C = s_f[1], s_f[2], s_f[3]
            else:
                # fallback: assume channels last if last dim small
                if s_f[-1] in (1,3):
                    self.is_nchw = False
                    self.H, self.W, self.C = s_f[1], s_f[2], s_f[3]
                else:
                    self.is_nchw = True
                    self.C, self.H, self.W = s_f[1], s_f[2], s_f[3]
        elif len(s_f) == 3:
            # [C,H,W] or [H,W,C]
            if s_f[0] == 3:
                self.is_nchw = True
                self.C, self.H, self.W = s_f[0], s_f[1], s_f[2]
            elif s_f[-1] == 3:
                self.is_nchw = False
                self.H, self.W, self.C = s_f[0], s_f[1], s_f[2]
            else:
                self.H, self.W, self.C = 224, 224, 3
        else:
            self.H, self.W, self.C = 224, 224, 3

        # fallback numeric values if None or -1
        if self.H <= 0 or self.W <= 0:
            self.H, self.W = 224, 224
        if self.C not in (1,3):
            self.C = 3

    def preprocess(self, img):
        """Convert BGR img -> model input (float32)."""
        # convert BGR->RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize
        img_r = cv2.resize(img_rgb, (int(self.W), int(self.H)))
        x = img_r.astype("float32") / 255.0
        if self.is_nchw:
            x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
            x = np.expand_dims(x, axis=0).astype("float32")
        else:
            x = np.expand_dims(x, axis=0).astype("float32")
        return x

    def predict(self, img):
        """
        img: BGR crop (numpy)
        returns: (prob_mask, prob_nomask)
        """
        x = self.preprocess(img)
        out = self.sess.run([self.output_name], {self.input_name: x})
        out = out[0]
        # try to produce two-class softmaxed probs
        if out.ndim == 2 and out.shape[1] >= 2:
            scores = out[0]
            probs = softmax(scores)
            return float(probs[0]), float(probs[1])
        else:
            flat = out.flatten()
            probs = softmax(flat)
            if probs.size >= 2:
                return float(probs[0]), float(probs[1])
            elif probs.size == 1:
                p = float(probs[0])
                return p, 1.0 - p
            else:
                raise RuntimeError("Unexpected ONNX output shape for mask model.")


def softmax(x):
    a = np.array(x, dtype=np.float32)
    e = np.exp(a - np.max(a))
    s = e / e.sum()
    return s


# -------------------------
# Calibration & detectors (similar to your script)
# -------------------------
class Calibrator:
    def __init__(self, save_path="calib.json"):
        self.image_points = []
        self.world_points = []
        self.H = None
        self.save_path = save_path

    def click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.image_points) < 4:
            self.image_points.append((int(x), int(y)))
            print("Clicked:", (x, y))

    def calibrate_interactive(self, frame):
        self.image_points = []
        win = "Calibration - click 4 points"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.click_callback)
        print("Click 4 points on the ground plane (order matters). Press 'q' to cancel.")
        while True:
            disp = frame.copy()
            for p in self.image_points:
                cv2.circle(disp, (p[0], p[1]), 6, (0,255,0), -1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if len(self.image_points) >= 4:
                break
            if k == ord('q'):
                break
        cv2.destroyWindow(win)
        if len(self.image_points) < 4:
            print("Calibration cancelled.")
            return False
        self.world_points = []
        for i in range(4):
            while True:
                s = input(f"Point {i+1} (x,y in meters): ").strip()
                try:
                    xw,yw = map(float, s.split(","))
                    self.world_points.append((xw,yw))
                    break
                except:
                    print("Invalid, use x,y (e.g. 0,0)")
        img_pts = np.array(self.image_points, dtype=np.float32)
        wrld_pts = np.array(self.world_points, dtype=np.float32)
        H, status = cv2.findHomography(img_pts, wrld_pts)
        if H is None:
            print("Failed to compute homography.")
            return False
        self.H = H
        return True

    def save(self):
        data = {"image_points": self.image_points, "world_points": self.world_points, "H": self.H.tolist() if self.H is not None else None}
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved calibration to", self.save_path)

    def load(self):
        if not os.path.isfile(self.save_path):
            print("No calibration file:", self.save_path)
            return False
        d = json.load(open(self.save_path))
        self.image_points = [tuple(p) for p in d.get("image_points",[])]
        self.world_points = [tuple(p) for p in d.get("world_points",[])]
        H = d.get("H", None)
        if H is not None:
            self.H = np.array(H)
        return True


# -------------------------
# Detection helpers
# -------------------------
def detect_people(frame, net, conf_thresh=0.5):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300,300), mean=(127.5,127.5,127.5), swapRB=False, crop=False)
    net.setInput(blob)
    dets = net.forward()
    people = []
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < conf_thresh:
            continue
        idx = int(dets[0,0,i,1])
        if idx < 0 or idx >= len(CLASSES):
            continue
        if CLASSES[idx] != "person":
            continue
        box = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
        sx,sy,ex,ey = box
        sx=max(0,sx); sy=max(0,sy); ex=min(w-1,ex); ey=min(h-1,ey)
        people.append({"bbox":(sx,sy,ex,ey), "bottom_center":(int((sx+ex)/2), ey), "confidence":conf, "coords":None, "violation":False, "mask":None})
    return people

def detect_face_in_person(frame, person, face_net, conf_thresh=0.5):
    sx,sy,ex,ey = person["bbox"]
    fy1 = sy
    fy2 = sy + (ey - sy)//2
    roi = frame[fy1:fy2, sx:ex]
    if roi.size == 0:
        return None
    h,w = roi.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    dets = face_net.forward()
    best = None; best_c = 0.0
    for i in range(dets.shape[2]):
        c = float(dets[0,0,i,2])
        if c > conf_thresh and c > best_c:
            box = (dets[0,0,i,3:7] * np.array([w,h,w,h])).astype(int)
            best = box; best_c = c
    if best is None:
        return None
    fx1, fy1b, fx2, fy2b = best
    fx1_abs = sx + max(0, fx1); fx2_abs = sx + min(w, fx2)
    fy1_abs = fy1 + max(0, fy1b); fy2_abs = fy1 + min(h, fy2b)
    crop = frame[fy1_abs:fy2_abs, fx1_abs:fx2_abs]
    return crop


def draw(frame, people, fps=None):
    for i,p in enumerate(people, start=1):
        sx,sy,ex,ey = p["bbox"]
        # determine color: red for any violation
        any_violation = p.get("violation_distance", False) or p.get("violation_mask", False)
        color = (0,0,255) if any_violation else (0,255,0)
        cv2.rectangle(frame, (sx,sy),(ex,ey), color, 2)
        label = f"ID {i} {p['confidence']:.2f}"
        cv2.putText(frame, label, (sx, max(15,sy-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # status text priority: both -> "Violates: Mask & Distance"
        vd = p.get("violation_distance", False)
        vm = p.get("violation_mask", False)
        status = "Safe"
        if vd and vm:
            status = "Violates: Mask & Distance"
        elif vd:
            status = "Violation: Distance"
        elif vm:
            status = "Violation: NoMask"
        cv2.putText(frame, status, (sx, sy-26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255) if any_violation else (0,200,0), 2)

        # if mask probabilities present, show small text
        if p.get("mask") is not None:
            pm, pnm = p["mask"]
            cv2.putText(frame, f"M:{pm:.2f} N:{pnm:.2f}", (sx, ey+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"Violations(total people flagged): {sum(1 for pp in people if pp.get('violation_mask') or pp.get('violation_distance'))}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--proto", default=MOBILENET_PROTO)
    p.add_argument("--model", default=MOBILENET_MODEL)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--min-distance", type=float, default=2.0)
    p.add_argument("--calib", default="calib.json")
    p.add_argument("--load-calib", action="store_true")
    p.add_argument("--no-save-calib", action="store_true")
    p.add_argument("--show-birdeye", action="store_true")
    p.add_argument("--face-detect", action="store_true", help="Enable face detection for mask classification")
    p.add_argument("--mask-onnx", default=None, help="Path to ONNX mask classifier (optional)")
    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # ensure mobilenet files
    try:
        ensure_file(args.proto, MOBILENET_PROTO_URL)
        ensure_file(args.model, MOBILENET_MODEL_URL)
    except Exception as e:
        print("Failed to ensure MobileNetSSD:", e)
        sys.exit(1)

    # load person detector
    try:
        net = cv2.dnn.readNetFromCaffe(args.proto, args.model)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Loaded MobileNet-SSD.")
    except Exception as e:
        print("Failed to load MobileNet-SSD:", e)
        sys.exit(1)

    # face & mask
    face_net = None
    mask_clf = None
    if args.face_detect or args.mask_onnx:
        try:
            ensure_file(FACE_PROTO, FACE_PROTO_URL)
            # try multiple mirrors for model
            try:
                ensure_file(FACE_MODEL, FACE_MODEL_URLS[0])
            except Exception:
                ensure_file(FACE_MODEL, FACE_MODEL_URLS[1])
            face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Loaded face detector.")
        except Exception as e:
            print("Warning: could not load face detector:", e)
            face_net = None

        if args.mask_onnx:
            if ort is None:
                print("ERROR: onnxruntime not installed. pip install onnxruntime")
                sys.exit(1)
            if not os.path.isfile(args.mask_onnx):
                print("ERROR: mask ONNX not found at", args.mask_onnx)
                sys.exit(1)
            try:
                mask_clf = ONNXMaskClassifier(args.mask_onnx)
            except Exception as e:
                print("ERROR: failed to load ONNX mask model:", e)
                sys.exit(1)

    # open capture
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("ERROR: cannot open video source")
        sys.exit(1)
    ret, frame0 = cap.read()
    if not ret:
        print("ERROR: cannot read from source")
        sys.exit(1)

    # calibration
    calib = Calibrator(save_path=args.calib)
    if args.load_calib:
        ok = calib.load()
        if not ok or calib.H is None:
            print("ERROR: failed to load calibration. Provide a valid calib.json or remove --load-calib")
            cap.release()
            sys.exit(1)
    else:
        print("Interactive calibration starting...")
        ok = calib.calibrate_interactive(frame0)
        if not ok:
            print("Calibration failed. Exiting.")
            cap.release()
            sys.exit(1)
        if not args.no_save_calib:
            calib.save()

    H = calib.H

    # birds-eye canvas
    birdeye_canvas = None
    if args.show_birdeye and calib.world_points:
        wrld = np.array(calib.world_points)
        minx,miny = wrld.min(axis=0); maxx,maxy = wrld.max(axis=0)
        scale = 150
        W = int((maxx-minx)*scale)+200
        Hc = int((maxy-miny)*scale)+200
        birdeye_canvas = np.zeros((Hc, W, 3), dtype=np.uint8)

    fps_smooth = None
    last_t = time.time()

    cv2.namedWindow("Social Distance Monitor", cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty("Social Distance Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except:
        pass

    print("Starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        people = detect_people(frame, net, conf_thresh=args.conf_thresh)

        # world coords
        if len(people) > 0:
            pts = [p["bottom_center"] for p in people]
            try:
                wrld = compute_world_coords(pts, H)
                for i,coord in enumerate(wrld):
                    people[i]["coords"] = coord
            except Exception as e:
                print("WARN: failed compute world coords:", e)
                for p in people:
                    p["coords"] = None

        # reset violation flags
        for p in people:
            p["violation_distance"] = False
            p["violation_mask"] = False

        # pairwise distance check
        for i in range(len(people)):
            for j in range(i+1, len(people)):
                if people[i]["coords"] is None or people[j]["coords"] is None:
                    continue
                xi,yi = people[i]["coords"]; xj,yj = people[j]["coords"]
                d = math.hypot(xi-xj, yi-yj)
                if d < args.min_distance:
                    people[i]["violation_distance"] = True
                    people[j]["violation_distance"] = True

        # face + mask classification
        if mask_clf is not None and face_net is not None:
            for p in people:
                try:
                    crop = detect_face_in_person(frame, p, face_net, conf_thresh=0.5)
                    if crop is None or crop.size == 0:
                        p["mask"] = None
                        p["violation_mask"] = False
                        continue
                    pm, pnm = mask_clf.predict(crop)  # returns (prob_mask, prob_nomask)
                    p["mask"] = (pm, pnm)
                    # decide mask: if pm >= pnm => masked; else no-mask
                    if pm >= pnm:
                        p["violation_mask"] = False
                    else:
                        p["violation_mask"] = True
                except Exception as e:
                    # don't crash on ONNX errors; set no mask info
                    print("Mask classification failed:", e)
                    p["mask"] = None
                    p["violation_mask"] = False

        # final combined label: any violation if distance or mask violation
        for p in people:
            p["violation"] = p.get("violation_distance", False) or p.get("violation_mask", False)

        # draw, show
        dt = time.time() - last_t
        last_t = time.time()
        fps_inst = 1.0/dt if dt>0 else 0.0
        fps_smooth = fps_inst if fps_smooth is None else 0.9*fps_smooth + 0.1*fps_inst

        draw(frame, people, fps=fps_smooth)
        cv2.imshow("Social Distance Monitor", frame)

        # birds-eye
        if args.show_birdeye and birdeye_canvas is not None and calib.world_points:
            birdeye_canvas[:] = 0
            wrld = np.array(calib.world_points)
            minx,miny = wrld.min(axis=0); maxx,maxy = wrld.max(axis=0)
            scale = 150
            wrld_to_canvas = lambda x,y: (int((x-minx)*scale)+50, int((y-miny)*scale)+50)
            if calib.world_points:
                pts = np.array([wrld_to_canvas(x,y) for (x,y) in calib.world_points], dtype=np.int32)
                cv2.polylines(birdeye_canvas, [pts], isClosed=True, color=(100,100,100), thickness=1)
            for p in people:
                if p["coords"] is None:
                    continue
                cx,cy = wrld_to_canvas(p["coords"][0], p["coords"][1])
                col = (0,0,255) if p["violation"] else (0,255,0)
                cv2.circle(birdeye_canvas, (cx,cy), 6, col, -1)
            cv2.imshow("Birds-eye view (meters scaled)", birdeye_canvas)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting.")


def compute_world_coords(points_image, H):
    pts = np.array(points_image, dtype=np.float32).reshape(-1,1,2)
    wrld = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    return wrld


if __name__ == "__main__":
    main()
# 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