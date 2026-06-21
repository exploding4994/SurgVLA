import dataclasses
import numpy as np
import einops
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """解析图像为uint8格式(H,W,C)"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # C,H,W -> H,W,C
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MyInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. 解析图像
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 2. 构建状态（如果有多个状态，需要拼接）
        state = data["observation/state"]

        # 3. 创建输入字典（键名必须匹配模型期望）
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 填充缺失图像
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if self.model_type != _model.ModelType.PI0_FAST else np.True_,
            },
        }

        # 4. 添加动作（仅训练时）
        if "actions" in data:
            inputs["actions"] = data["actions"]

            # 5. 添加提示
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class MyOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
            # 返回前N个动作维度，去除填充
        return {"actions": np.asarray(data["actions"][:, :7])}