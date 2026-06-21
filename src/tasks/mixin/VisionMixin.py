import time

import cv2
import numpy as np

from ok import BaseTask


class VisionMixin(BaseTask):
    _rotated_template_cache = {}

    def _find_rotated_template(
        self,
        feature_name,
        scene: np.ndarray,
        threshold=0.75,
        angle_range=range(-180, 180, 2),
        min_non_zero=20,
        template_angle=0,
    ):
        start_time = time.time()
        template = self.get_feature_by_name(feature_name).mat
        scene_mask = self._first_channel_mask(scene)
        if cv2.countNonZero(scene_mask) < min_non_zero:
            return [], (time.time() - start_time) * 1000

        best = None
        for angle, rotated_template in self._get_rotated_templates(
            template,
            angle_range=angle_range,
            min_non_zero=min_non_zero,
            cache_key=feature_name,
        ):
            th, tw = rotated_template.shape[:2]
            if th > scene_mask.shape[0] or tw > scene_mask.shape[1]:
                continue

            result = cv2.matchTemplate(scene_mask, rotated_template, cv2.TM_CCOEFF_NORMED)
            _, score, _, top_left = cv2.minMaxLoc(result)
            if best is None or score > best["score"]:
                best = {
                    "center": (top_left[0] + tw // 2, top_left[1] + th // 2),
                    "angle": self._normalize_angle(angle + template_angle),
                    "match_angle": angle,
                    "score": score,
                }

        if best is None or best["score"] < threshold:
            return [], (time.time() - start_time) * 1000

        best["score"] = round(best["score"], 3)
        return [best], (time.time() - start_time) * 1000

    def _get_rotated_templates(
        self,
        template: np.ndarray,
        angle_range=range(-180, 180, 5),
        min_non_zero=20,
        cache_key=None,
    ):
        template_mask = self._trim_mask(self._first_channel_mask(template))
        angles = tuple(angle_range)
        template_key = (
            cache_key or id(template),
            template_mask.shape,
            cv2.countNonZero(template_mask),
            hash(template_mask.tobytes()),
            angles,
            min_non_zero,
        )
        cached = self._rotated_template_cache.get(template_key)
        if cached is not None:
            return cached

        templates = []
        for angle in angles:
            rotated = self._rotate_mask(template_mask, angle)
            rotated = self._trim_mask(rotated)
            if cv2.countNonZero(rotated) >= min_non_zero:
                templates.append((angle, rotated))

        self._rotated_template_cache[template_key] = templates
        return templates

    @staticmethod
    def _first_channel_mask(mat: np.ndarray):
        if mat.ndim == 2:
            return mat
        return mat[:, :, 0]

    @staticmethod
    def _normalize_angle(angle):
        return (angle + 180) % 360 - 180

    def _rotate_mask(self, mask: np.ndarray, angle):
        h, w = mask.shape[:2]
        center = (w / 2, h / 2)
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(rotate_matrix[0, 0])
        sin = abs(rotate_matrix[0, 1])
        new_w = int(round(h * sin + w * cos))
        new_h = int(round(h * cos + w * sin))
        rotate_matrix[0, 2] += new_w / 2 - center[0]
        rotate_matrix[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(
            mask,
            rotate_matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )

    def _trim_mask(self, mask):
        points = cv2.findNonZero(mask)
        if points is None:
            return mask
        x, y, w, h = cv2.boundingRect(points)
        return mask[y : y + h, x : x + w]

    def _find_contours_from_first_channel(self, bgr):
        bin_mat = bgr[:, :, 0]
        contours, _ = cv2.findContours(bin_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _find_rotated_shape(self, target_contour, scene_contours, score_threshold=0.1):
        """
        target_contour: 要匹配的目标轮廓。
        scene_contours: 在场景中找到的候选轮廓。
        score_threshold: 越小越严格。通常 0.05-0.2 之间。
        """
        start_time = time.time()

        results = []
        for cnt in scene_contours:
            if cv2.contourArea(cnt) < 50:
                continue

            # 核心算法：比较两个形状的胡氏矩 (I1 模式最常用)
            # 返回值越小，匹配度越高（0 为完美匹配）
            score = cv2.matchShapes(target_contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)

            if score < score_threshold:
                # 计算重心和角度
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 使用最小外接矩形获取角度
                    rect = cv2.minAreaRect(cnt)
                    angle = rect[2]  # 得到角度

                    results.append({"center": (cx, cy), "angle": angle, "score": round(score, 3)})

        # 按分数升序排列（得分越低越好）
        results = sorted(results, key=lambda x: x["score"])
        return results, (time.time() - start_time) * 1000
