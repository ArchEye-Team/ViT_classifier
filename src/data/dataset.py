from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)


class Dataset(TorchDataset):
    def __init__(
        self, data_module, split,
        mean_value=None, std_value=None,
    ):
        self.data_module = data_module
        self.data_split = split
        
        self._mean_value = mean_value if mean_value is not None else [0.485, 0.456, 0.406]
        self._std_value = std_value if std_value is not None else [0.229, 0.224, 0.225]

        self.num_classes = 0
        self.class2id = {}
        self.id2class = []
        self.class_count = []

        self.dataset = []

        dataset_dir = self.data_module.data_path
        class_dirs = sorted((class_dir for class_dir in dataset_dir.iterdir() if class_dir.is_dir()),
                            key=lambda class_dir: class_dir.name)

        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.class2id:
                self.class2id[class_name] = self.num_classes
                self.id2class.append(class_name)
                self.class_count.append(0)
                self.num_classes += 1

            class_id = self.num_classes - 1

            labeled_images = [(image_path, class_id) for image_path in
                              sorted((image_path for image_path in class_dir.iterdir() if image_path.is_file()),
                                     key=lambda image_path: image_path.name)]

            self.class_count[class_id] += len(labeled_images)
            
            class_ids = [val[1] for val in labeled_images]
            train, test = train_test_split(labeled_images, test_size=0.2, stratify=class_ids)
            class_ids = [val[1] for val in train]
            train, val = train_test_split(train, test_size=0.2, stratify=class_ids)
            
            match self.data_split:
                case 'train':
                    self.dataset.extend(train)
                case 'val':
                    self.dataset.extend(val)
                case 'test':
                    self.dataset.extend(test)
                case '_':
                    raise ValueError(f'Unsupported value of split argument: {self.data_split}')

        self.class_weights = [1 - count / sum(self.class_count) for count in self.class_count]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, class_id = self.dataset[index]

        image = Image.open(str(image_path)).convert('RGB')

        transforms = []
        
        if self.data_split == 'train':
            transforms.extend([
                RandomResizedCrop(self.data_module.image_size),
                RandomHorizontalFlip(),
            ])
        else:
            transforms.extend([
                Resize(self.data_module.image_size),
                CenterCrop(self.data_module.image_size),
            ])

        transforms.extend([
            ToTensor(),
            Normalize(mean=self._mean_value, std=self._std_value)
        ])

        image = Compose(transforms)(image)

        return image, class_id
