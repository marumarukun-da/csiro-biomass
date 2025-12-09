import os

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_train_transform_from_config(aug_cfg: dict, img_size: int, in_chans: int = 3) -> A.Compose:
    """
    augmentation設定からトレーニング用のtransformを構築する。

    Args:
        aug_cfg: augmentationセクションの設定
        img_size: 画像サイズ
        in_chans: 入力チャンネル数（1: グレースケール, 3: RGB）

    Returns:
        albumentationsのCompose
    """
    # Normalize設定の取得
    normalize_cfg = aug_cfg.get("normalize", {})
    if in_chans == 1:
        mean = tuple(normalize_cfg.get("grayscale", {}).get("mean", [0.456]))
        std = tuple(normalize_cfg.get("grayscale", {}).get("std", [0.224]))
    else:
        mean = tuple(normalize_cfg.get("rgb", {}).get("mean", [0.485, 0.456, 0.406]))
        std = tuple(normalize_cfg.get("rgb", {}).get("std", [0.229, 0.224, 0.225]))

    # トレーニング用augmentation設定の取得
    train_cfg = aug_cfg.get("train", {})

    # augmentationリストの構築
    transforms = []

    # Resizeは常に追加
    transforms.append(A.Resize(img_size, img_size))

    # HorizontalFlip
    h_flip_cfg = train_cfg.get("horizontal_flip", {})
    if isinstance(h_flip_cfg, dict):
        h_flip_p = h_flip_cfg.get("p", 0.0)
        if h_flip_p > 0:
            transforms.append(A.HorizontalFlip(p=h_flip_p))

    # VerticalFlip
    v_flip_cfg = train_cfg.get("vertical_flip", {})
    if isinstance(v_flip_cfg, dict):
        v_flip_p = v_flip_cfg.get("p", 0.0)
        if v_flip_p > 0:
            transforms.append(A.VerticalFlip(p=v_flip_p))

    # RandomBrightnessContrast
    rbc_cfg = train_cfg.get("random_brightness_contrast", {})
    if isinstance(rbc_cfg, dict):
        rbc_p = rbc_cfg.get("p", 0.0)
        if rbc_p > 0:
            brightness_limit = rbc_cfg.get("brightness_limit", 0.2)
            contrast_limit = rbc_cfg.get("contrast_limit", 0.2)
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=rbc_p,
                )
            )

    # CoarseDropout
    coarse_dropout_cfg = train_cfg.get("coarse_dropout", {})
    if isinstance(coarse_dropout_cfg, dict) and coarse_dropout_cfg.get("enabled", False):
        cd_p = coarse_dropout_cfg.get("p", 0.0)
        if cd_p > 0:
            num_holes_range = tuple(coarse_dropout_cfg.get("num_holes_range", [10, 30]))
            hole_height_range = tuple(coarse_dropout_cfg.get("hole_height_range", [8, 16]))
            hole_width_range = tuple(coarse_dropout_cfg.get("hole_width_range", [8, 16]))
            transforms.append(
                A.CoarseDropout(
                    num_holes_range=num_holes_range,
                    hole_height_range=hole_height_range,
                    hole_width_range=hole_width_range,
                    p=cd_p,
                )
            )

    # Normalize と ToTensorV2 は常に追加
    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_valid_transform_from_config(aug_cfg: dict, img_size: int, in_chans: int = 3) -> A.Compose:
    """
    augmentation設定からバリデーション用のtransformを構築する。

    Args:
        aug_cfg: augmentationセクションの設定
        img_size: 画像サイズ
        in_chans: 入力チャンネル数（1: グレースケール, 3: RGB）

    Returns:
        albumentationsのCompose
    """
    # Normalize設定の取得
    normalize_cfg = aug_cfg.get("normalize", {})
    if in_chans == 1:
        mean = tuple(normalize_cfg.get("grayscale", {}).get("mean", [0.456]))
        std = tuple(normalize_cfg.get("grayscale", {}).get("std", [0.224]))
    else:
        mean = tuple(normalize_cfg.get("rgb", {}).get("mean", [0.485, 0.456, 0.406]))
        std = tuple(normalize_cfg.get("rgb", {}).get("std", [0.229, 0.224, 0.225]))

    # バリデーションはResize + Normalize + ToTensorV2のみ
    transforms = [
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


class SampleDataset(Dataset):
    """
    画像分類タスク用の汎用データセットクラス

    Args:
        image_paths: 画像ファイルのパスリスト
        labels: 各画像のラベル（分類タスクの場合）
        transform: 画像に適用する変換処理
        class_names: クラス名のリスト（オプション）
        in_chans: 入力チャンネル数（1: グレースケール, 3: RGB）
    """

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int] | None = None,
        transform: A.Compose | None = None,
        class_names: list[str] | None = None,
        in_chans: int = 3,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names
        self.in_chans = in_chans

        # データの整合性チェック
        if self.labels is not None and len(self.image_paths) != len(self.labels):
            raise ValueError(f"image_pathsとlabelsの長さが一致しません: {len(self.image_paths)} != {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # 画像の読み込み
        image_path = self.image_paths[idx]

        if self.in_chans == 1:
            # グレースケールで読み込み
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"画像が読み込めません: {image_path}")

        else:
            # RGB（3チャンネル）で読み込み
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"画像が読み込めません: {image_path}")
            # BGR -> RGB変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 変換処理の適用
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # 返り値の構築
        item = {"image": image, "image_path": image_path}

        if self.labels is not None:
            item["label"] = self.labels[idx]

        return item

    def get_class_name(self, label: int) -> str | None:
        """ラベルからクラス名を取得"""
        if self.class_names is not None and 0 <= label < len(self.class_names):
            return self.class_names[label]
        return None

    @classmethod
    def from_directory(
        cls,
        root_dir: str,
        transform: A.Compose | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        in_chans: int = 3,
    ) -> "SampleDataset":
        """
        ディレクトリ構造からデータセットを作成

        ディレクトリ構造:
        root_dir/
            class_0/
                image1.jpg
                image2.jpg
            class_1/
                image3.jpg

        Args:
            root_dir: ルートディレクトリパス
            transform: 画像変換処理
            extensions: 画像ファイルの拡張子
            in_chans: 入力チャンネル数（1: グレースケール, 3: RGB）
        """
        image_paths = []
        labels = []
        class_names = []

        # クラスディレクトリの取得（ソート済み）
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for label, class_name in enumerate(class_dirs):
            class_names.append(class_name)
            class_dir = os.path.join(root_dir, class_name)

            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(extensions):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(label)

        return cls(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            class_names=class_names,
            in_chans=in_chans,
        )
