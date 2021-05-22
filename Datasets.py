import os
from torchvision import transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torch
import Config as cfg


class ClassificationDataset:
    def __init__(self, data_dir, class_labels, shape, testset_size, trainset_size, expected_files):
        # Basic Dataset Info
        self._class_labels = tuple(class_labels)
        self._shape = tuple(shape)
        self._testset_size = testset_size
        self._trainset_size = trainset_size
        self._data_dir = data_dir

        if not isinstance(expected_files, list):
            self._expected_files = [expected_files]
        else:
            self._expected_files = expected_files

        self._download = True if any(
            not os.path.isfile(os.path.join(self._data_dir, file)) for file in self._expected_files) else False

    def name(self):
        assert self.__class__.__name__ != 'ClassificationDataset'
        return self.__class__.__name__

    def num_classes(self):
        return len(self._class_labels)

    def class_labels(self):
        return self._class_labels

    def input_channels(self):
        return self._shape[0]

    def shape(self):
        return self._shape

    def max_test_size(self):
        return self._testset_size

    def max_train_size(self):
        return self._trainset_size

    def testset(self, batch_size, max_samples=None, device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 32 if not cfg.DEBUG else 0, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4 if not cfg.DEBUG else 0, False

        test_dataset = self._test_importer()

        if max_samples is None or max_samples >= self._testset_size:
            testset_sz = self._testset_size
            test_sampler = None
        else:
            testset_sz = max_samples
            test_sampler = SequentialSampler(list(range(max_samples)))

        test_gen = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

        return test_gen, testset_sz

    def trainset(self, batch_size=128, valid_size=0.0, max_samples=None, augment=True, shuffle=True, random_seed=None,
                 device='cuda'):

        if device.lower() == 'cuda' and torch.cuda.is_available():
            num_workers, pin_memory = 32 if not cfg.DEBUG else 0, True
        else:
            print('Warning: Did not find working GPU - Loading dataset on CPU')
            num_workers, pin_memory = 4 if not cfg.DEBUG else 0, False

        max_samples = self._trainset_size if max_samples is None else min(self._trainset_size, max_samples)
        assert ((valid_size >= 0) and (valid_size <= 1)), "Valid_size should be in the range [0, 1]."

        train_dataset = self._train_importer(augment)
        val_dataset = self._train_importer(False)       # Don't augment validation

        indices = list(range(self._trainset_size))
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)

        indices = indices[:max_samples]  # Truncate to desired size
        split = int(np.floor(valid_size * max_samples))  # Split validation
        train_ids, valid_ids = indices[split:], indices[:split]

        num_train = len(train_ids)
        num_valid = len(valid_ids)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(train_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_ids), num_workers=num_workers,
                                                   pin_memory=pin_memory)

        return (train_loader, num_train), (valid_loader, num_valid)

    def _train_importer(self, augment=True):
        raise NotImplementedError

    def _test_importer(self):
        raise NotImplementedError


class CIFAR10(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(
            data_dir=data_dir,
            class_labels=CIFAR10_NAMES,
            shape=(3, 32, 32),
            testset_size=10000,
            trainset_size=50000,
            expected_files=os.path.join('CIFAR10', 'cifar-10-python.tar.gz')
        )

    def _train_importer(self, augment=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if augment:
            ops = [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize]
        else:
            ops = [transforms.ToTensor(),
                   normalize]

        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=True, download=self._download,
                                 transform=transforms.Compose(ops))

    def _test_importer(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ops = [transforms.ToTensor(), normalize]

        return datasets.CIFAR10(root=os.path.join(self._data_dir, 'CIFAR10'), train=False,
                                 download=self._download,
                                 transform=transforms.Compose(ops))


class CIFAR100(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(
            data_dir=data_dir,
            class_labels=CIFAR100_NAMES,
            shape=(3, 32, 32),
            testset_size=10000,
            trainset_size=50000,
            expected_files=os.path.join('CIFAR100', 'cifar-100-python.tar.gz')
        )

    def _train_importer(self, augment=True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if augment:
            ops = [transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize]
        else:
            ops = [transforms.ToTensor(),
                               normalize]

        return datasets.CIFAR100(root=os.path.join(self._data_dir, 'CIFAR100'), train=True, download=self._download,
                                 transform=transforms.Compose(ops))

    def _test_importer(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ops = [transforms.ToTensor(), normalize]

        return datasets.CIFAR100(root=os.path.join(self._data_dir, 'CIFAR100'), train=False, download=self._download,
                                 transform=transforms.Compose(ops))


class ImageNet(ClassificationDataset):
    def __init__(self, data_dir):
        super().__init__(data_dir=data_dir,
                         class_labels=IMAGENET_LABEL_NAMES,
                         shape=(3, 224, 224),
                         testset_size=50000,
                         trainset_size=1200000,
                         expected_files=[os.path.join('ImageNet', 'train'), os.path.join('ImageNet', 'val')])

    def _train_importer(self, augment=True):
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if augment:
            ops = [transforms.RandomResizedCrop(224 if not cfg.INCEPTION else 299),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   normalize]
        else:
            ops = [transforms.Resize(256 if not cfg.INCEPTION else 342),
                   transforms.CenterCrop(224 if not cfg.INCEPTION else 299),
                   transforms.ToTensor(),
                   normalize]

        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'train'),
                                    transform=transforms.Compose(ops))

    def _test_importer(self):
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ops = [transforms.Resize(256 if not cfg.INCEPTION else 342),
               transforms.CenterCrop(224 if not cfg.INCEPTION else 299),
               transforms.ToTensor(),
               normalize]

        return datasets.ImageFolder(root=os.path.join(self._data_dir, 'val'),
                                    transform=transforms.Compose(ops))


class Datasets:
    @staticmethod
    def which():
        return tuple(cfg.DATASETS.keys())

    @staticmethod
    def get(dataset_name):
        dataset_ptr = cfg.DATASETS[dataset_name]['ptr']
        dataset_dir = cfg.DATASETS[dataset_name]['dir']
        return dataset_ptr(dataset_dir)


CIFAR10_NAMES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


CIFAR100_NAMES = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                  'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                  'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                  'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                  'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                  'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                  'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                  'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                  'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                  'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
                  'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
                  'woman', 'worm')

IMAGENET_LABEL_NAMES = ('kit_fox', 'english_setter', 'siberian_husky', 'australian_terrier',
                        'english_springer', 'grey_whale', 'lesser_panda', 'egyptian_cat', 'ibex',
                        'persian_cat', 'cougar', 'gazelle', 'porcupine', 'sea_lion', 'malamute',
                        'badger', 'great_dane', 'walker_hound', 'welsh_springer_spaniel', 'whippet',
                        'scottish_deerhound', 'killer_whale', 'mink', 'african_elephant', 'weimaraner',
                        'soft_coated_wheaten_terrier', 'dandie_dinmont', 'red_wolf',
                        'old_english_sheepdog', 'jaguar', 'otterhound', 'bloodhound', 'airedale',
                        'hyena', 'meerkat', 'giant_schnauzer', 'titi', 'three_toed_sloth', 'sorrel',
                        'black_footed_ferret', 'dalmatian', 'black_and_tan_coonhound', 'papillon',
                        'skunk', 'staffordshire_bullterrier', 'mexican_hairless',
                        'bouvier_des_flandres', 'weasel', 'miniature_poodle', 'cardigan', 'malinois',
                        'bighorn', 'fox_squirrel', 'colobus', 'tiger_cat', 'lhasa', 'impala', 'coyote',
                        'yorkshire_terrier', 'newfoundland', 'brown_bear', 'red_fox',
                        'norwegian_elkhound', 'rottweiler', 'hartebeest', 'saluki', 'grey_fox',
                        'schipperke', 'pekinese', 'brabancon_griffon', 'west_highland_white_terrier',
                        'sealyham_terrier', 'guenon', 'mongoose', 'indri', 'tiger', 'irish_wolfhound',
                        'wild_boar', 'entlebucher', 'zebra', 'ram', 'french_bulldog', 'orangutan',
                        'basenji', 'leopard', 'bernese_mountain_dog', 'maltese_dog', 'norfolk_terrier',
                        'toy_terrier', 'vizsla', 'cairn', 'squirrel_monkey', 'groenendael', 'clumber',
                        'siamese_cat', 'chimpanzee', 'komondor', 'afghan_hound', 'japanese_spaniel',
                        'proboscis_monkey', 'guinea_pig', 'white_wolf', 'ice_bear', 'gorilla', 'borzoi',
                        'toy_poodle', 'kerry_blue_terrier', 'ox', 'scotch_terrier', 'tibetan_mastiff',
                        'spider_monkey', 'doberman', 'boston_bull', 'greater_swiss_mountain_dog',
                        'appenzeller', 'shih_tzu', 'irish_water_spaniel', 'pomeranian',
                        'bedlington_terrier', 'warthog', 'arabian_camel', 'siamang',
                        'miniature_schnauzer', 'collie', 'golden_retriever', 'irish_terrier',
                        'affenpinscher', 'border_collie', 'hare', 'boxer', 'silky_terrier', 'beagle',
                        'leonberg', 'german_short_haired_pointer', 'patas', 'dhole', 'baboon',
                        'macaque', 'chesapeake_bay_retriever', 'bull_mastiff', 'kuvasz', 'capuchin',
                        'pug', 'curly_coated_retriever', 'norwich_terrier', 'flat_coated_retriever',
                        'hog', 'keeshond', 'eskimo_dog', 'brittany_spaniel', 'standard_poodle',
                        'lakeland_terrier', 'snow_leopard', 'gordon_setter', 'dingo',
                        'standard_schnauzer', 'hamster', 'tibetan_terrier', 'arctic_fox',
                        'wire_haired_fox_terrier', 'basset', 'water_buffalo', 'american_black_bear',
                        'angora', 'bison', 'howler_monkey', 'hippopotamus', 'chow', 'giant_panda',
                        'american_staffordshire_terrier', 'shetland_sheepdog', 'great_pyrenees',
                        'chihuahua', 'tabby', 'marmoset', 'labrador_retriever', 'saint_bernard',
                        'armadillo', 'samoyed', 'bluetick', 'redbone', 'polecat', 'marmot', 'kelpie',
                        'gibbon', 'llama', 'miniature_pinscher', 'wood_rabbit', 'italian_greyhound',
                        'lion', 'cocker_spaniel', 'irish_setter', 'dugong', 'indian_elephant', 'beaver',
                        'sussex_spaniel', 'pembroke', 'blenheim_spaniel', 'madagascar_cat',
                        'rhodesian_ridgeback', 'lynx', 'african_hunting_dog', 'langur', 'ibizan_hound',
                        'timber_wolf', 'cheetah', 'english_foxhound', 'briard', 'sloth_bear',
                        'border_terrier', 'german_shepherd', 'otter', 'koala', 'tusker', 'echidna',
                        'wallaby', 'platypus', 'wombat', 'revolver', 'umbrella', 'schooner',
                        'soccer_ball', 'accordion', 'ant', 'starfish', 'chambered_nautilus',
                        'grand_piano', 'laptop', 'strawberry', 'airliner', 'warplane', 'airship',
                        'balloon', 'space_shuttle', 'fireboat', 'gondola', 'speedboat', 'lifeboat',
                        'canoe', 'yawl', 'catamaran', 'trimaran', 'container_ship', 'liner', 'pirate',
                        'aircraft_carrier', 'submarine', 'wreck', 'half_track', 'tank', 'missile',
                        'bobsled', 'dogsled', 'bicycle_built_for_two', 'mountain_bike', 'freight_car',
                        'passenger_car', 'barrow', 'shopping_cart', 'motor_scooter', 'forklift',
                        'electric_locomotive', 'steam_locomotive', 'amphibian', 'ambulance',
                        'beach_wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'model_t',
                        'racer', 'sports_car', 'go_kart', 'golfcart', 'moped', 'snowplow',
                        'fire_engine', 'garbage_truck', 'pickup', 'tow_truck', 'trailer_truck',
                        'moving_van', 'police_van', 'recreational_vehicle', 'streetcar', 'snowmobile',
                        'tractor', 'mobile_home', 'tricycle', 'unicycle', 'horse_cart', 'jinrikisha',
                        'oxcart', 'bassinet', 'cradle', 'crib', 'four_poster', 'bookcase',
                        'china_cabinet', 'medicine_chest', 'chiffonier', 'table_lamp', 'file',
                        'park_bench', 'barber_chair', 'throne', 'folding_chair', 'rocking_chair',
                        'studio_couch', 'toilet_seat', 'desk', 'pool_table', 'dining_table',
                        'entertainment_center', 'wardrobe', 'granny_smith', 'orange', 'lemon', 'fig',
                        'pineapple', 'banana', 'jackfruit', 'custard_apple', 'pomegranate', 'acorn',
                        'hip', 'ear', 'rapeseed', 'corn', 'buckeye', 'organ', 'upright', 'chime',
                        'drum', 'gong', 'maraca', 'marimba', 'steel_drum', 'banjo', 'cello', 'violin',
                        'harp', 'acoustic_guitar', 'electric_guitar', 'cornet', 'french_horn',
                        'trombone', 'harmonica', 'ocarina', 'panpipe', 'bassoon', 'oboe', 'sax',
                        'flute', 'daisy', 'yellow_ladys_slipper', 'cliff', 'valley', 'alp', 'volcano',
                        'promontory', 'sandbar', 'coral_reef', 'lakeside', 'seashore', 'geyser',
                        'hatchet', 'cleaver', 'letter_opener', 'plane', 'power_drill', 'lawn_mower',
                        'hammer', 'corkscrew', 'can_opener', 'plunger', 'screwdriver', 'shovel', 'plow',
                        'chain_saw', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house_finch',
                        'junco', 'indigo_bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
                        'water_ouzel', 'kite', 'bald_eagle', 'vulture', 'great_grey_owl',
                        'black_grouse', 'ptarmigan', 'ruffed_grouse', 'prairie_chicken', 'peacock',
                        'quail', 'partridge', 'african_grey', 'macaw', 'sulphur_crested_cockatoo',
                        'lorikeet', 'coucal', 'bee_eater', 'hornbill', 'hummingbird', 'jacamar',
                        'toucan', 'drake', 'red_breasted_merganser', 'goose', 'black_swan',
                        'white_stork', 'black_stork', 'spoonbill', 'flamingo', 'american_egret',
                        'little_blue_heron', 'bittern', 'crane', 'limpkin', 'american_coot', 'bustard',
                        'ruddy_turnstone', 'red_backed_sandpiper', 'redshank', 'dowitcher',
                        'oystercatcher', 'european_gallinule', 'pelican', 'king_penguin', 'albatross',
                        'great_white_shark', 'tiger_shark', 'hammerhead', 'electric_ray', 'stingray',
                        'barracouta', 'coho', 'tench', 'goldfish', 'eel', 'rock_beauty', 'anemone_fish',
                        'lionfish', 'puffer', 'sturgeon', 'gar', 'loggerhead', 'leatherback_turtle',
                        'mud_turtle', 'terrapin', 'box_turtle', 'banded_gecko', 'common_iguana',
                        'american_chameleon', 'whiptail', 'agama', 'frilled_lizard', 'alligator_lizard',
                        'gila_monster', 'green_lizard', 'african_chameleon', 'komodo_dragon',
                        'triceratops', 'african_crocodile', 'american_alligator', 'thunder_snake',
                        'ringneck_snake', 'hognose_snake', 'green_snake', 'king_snake', 'garter_snake',
                        'water_snake', 'vine_snake', 'night_snake', 'boa_constrictor', 'rock_python',
                        'indian_cobra', 'green_mamba', 'sea_snake', 'horned_viper', 'diamondback',
                        'sidewinder', 'european_fire_salamander', 'common_newt', 'eft',
                        'spotted_salamander', 'axolotl', 'bullfrog', 'tree_frog', 'tailed_frog',
                        'whistle', 'wing', 'paintbrush', 'hand_blower', 'oxygen_mask', 'snorkel',
                        'loudspeaker', 'microphone', 'screen', 'mouse', 'electric_fan', 'oil_filter',
                        'strainer', 'space_heater', 'stove', 'guillotine', 'barometer', 'rule',
                        'odometer', 'scale', 'analog_clock', 'digital_clock', 'wall_clock', 'hourglass',
                        'sundial', 'parking_meter', 'stopwatch', 'digital_watch', 'stethoscope',
                        'syringe', 'magnetic_compass', 'binoculars', 'projector', 'sunglasses', 'loupe',
                        'radio_telescope', 'bow', 'cannon', 'assault_rifle', 'rifle', 'projectile',
                        'computer_keyboard', 'typewriter_keyboard', 'crane', 'lighter', 'abacus',
                        'cash_machine', 'slide_rule', 'desktop_computer', 'hand_held_computer',
                        'notebook', 'web_site', 'harvester', 'thresher', 'printer', 'slot',
                        'vending_machine', 'sewing_machine', 'joystick', 'switch', 'hook', 'car_wheel',
                        'paddlewheel', 'pinwheel', 'potters_wheel', 'gas_pump', 'carousel', 'swing',
                        'reel', 'radiator', 'puck', 'hard_disc', 'sunglass', 'pick', 'car_mirror',
                        'solar_dish', 'remote_control', 'disk_brake', 'buckle', 'hair_slide', 'knot',
                        'combination_lock', 'padlock', 'nail', 'safety_pin', 'screw', 'muzzle',
                        'seat_belt', 'ski', 'candle', 'jack_o_lantern', 'spotlight', 'torch',
                        'neck_brace', 'pier', 'tripod', 'maypole', 'mousetrap', 'spider_web',
                        'trilobite', 'harvestman', 'scorpion', 'black_and_gold_garden_spider',
                        'barn_spider', 'garden_spider', 'black_widow', 'tarantula', 'wolf_spider',
                        'tick', 'centipede', 'isopod', 'dungeness_crab', 'rock_crab', 'fiddler_crab',
                        'king_crab', 'american_lobster', 'spiny_lobster', 'crayfish', 'hermit_crab',
                        'tiger_beetle', 'ladybug', 'ground_beetle', 'long_horned_beetle', 'leaf_beetle',
                        'dung_beetle', 'rhinoceros_beetle', 'weevil', 'fly', 'bee', 'grasshopper',
                        'cricket', 'walking_stick', 'cockroach', 'mantis', 'cicada', 'leafhopper',
                        'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch',
                        'cabbage_butterfly', 'sulphur_butterfly', 'lycaenid', 'jellyfish',
                        'sea_anemone', 'brain_coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug',
                        'sea_slug', 'chiton', 'sea_urchin', 'sea_cucumber', 'iron', 'espresso_maker',
                        'microwave', 'dutch_oven', 'rotisserie', 'toaster', 'waffle_iron', 'vacuum',
                        'dishwasher', 'refrigerator', 'washer', 'crock_pot', 'frying_pan', 'wok',
                        'caldron', 'coffeepot', 'teapot', 'spatula', 'altar', 'triumphal_arch', 'patio',
                        'steel_arch_bridge', 'suspension_bridge', 'viaduct', 'barn', 'greenhouse',
                        'palace', 'monastery', 'library', 'apiary', 'boathouse', 'church', 'mosque',
                        'stupa', 'planetarium', 'restaurant', 'cinema', 'home_theater', 'lumbermill',
                        'coil', 'obelisk', 'totem_pole', 'castle', 'prison', 'grocery_store', 'bakery',
                        'barbershop', 'bookshop', 'butcher_shop', 'confectionery', 'shoe_shop',
                        'tobacco_shop', 'toyshop', 'fountain', 'cliff_dwelling', 'yurt', 'dock',
                        'brass', 'megalith', 'bannister', 'breakwater', 'dam', 'chainlink_fence',
                        'picket_fence', 'worm_fence', 'stone_wall', 'grille', 'sliding_door',
                        'turnstile', 'mountain_tent', 'scoreboard', 'honeycomb', 'plate_rack',
                        'pedestal', 'beacon', 'mashed_potato', 'bell_pepper', 'head_cabbage',
                        'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash', 'acorn_squash',
                        'butternut_squash', 'cucumber', 'artichoke', 'cardoon', 'mushroom',
                        'shower_curtain', 'jean', 'carton', 'handkerchief', 'sandal', 'ashcan', 'safe',
                        'plate', 'necklace', 'croquet_ball', 'fur_coat', 'thimble', 'pajama',
                        'running_shoe', 'cocktail_shaker', 'chest', 'manhole_cover', 'modem', 'tub',
                        'tray', 'balance_beam', 'bagel', 'prayer_rug', 'kimono', 'hot_pot',
                        'whiskey_jug', 'knee_pad', 'book_jacket', 'spindle', 'ski_mask', 'beer_bottle',
                        'crash_helmet', 'bottlecap', 'tile_roof', 'mask', 'maillot', 'petri_dish',
                        'football_helmet', 'bathing_cap', 'teddy', 'holster', 'pop_bottle',
                        'photocopier', 'vestment', 'crossword_puzzle', 'golf_ball', 'trifle', 'suit',
                        'water_tower', 'feather_boa', 'cloak', 'red_wine', 'drumstick', 'shield',
                        'christmas_stocking', 'hoopskirt', 'menu', 'stage', 'bonnet', 'meat_loaf',
                        'baseball', 'face_powder', 'scabbard', 'sunscreen', 'beer_glass',
                        'hen_of_the_woods', 'guacamole', 'lampshade', 'wool', 'hay', 'bow_tie',
                        'mailbag', 'water_jug', 'bucket', 'dishrag', 'soup_bowl', 'eggnog', 'mortar',
                        'trench_coat', 'paddle', 'chain', 'swab', 'mixing_bowl', 'potpie',
                        'wine_bottle', 'shoji', 'bulletproof_vest', 'drilling_platform', 'binder',
                        'cardigan', 'sweatshirt', 'pot', 'birdhouse', 'hamper', 'ping_pong_ball',
                        'pencil_box', 'pay_phone', 'consomme', 'apron', 'punching_bag', 'backpack',
                        'groom', 'bearskin', 'pencil_sharpener', 'broom', 'mosquito_net', 'abaya',
                        'mortarboard', 'poncho', 'crutch', 'polaroid_camera', 'space_bar', 'cup',
                        'racket', 'traffic_light', 'quill', 'radio', 'dough', 'cuirass',
                        'military_uniform', 'lipstick', 'shower_cap', 'monitor', 'oscilloscope',
                        'mitten', 'brassiere', 'french_loaf', 'vase', 'milk_can', 'rugby_ball',
                        'paper_towel', 'earthstar', 'envelope', 'miniskirt', 'cowboy_hat', 'trolleybus',
                        'perfume', 'bathtub', 'hotdog', 'coral_fungus', 'bullet_train', 'pillow',
                        'toilet_tissue', 'cassette', 'carpenters_kit', 'ladle', 'stinkhorn', 'lotion',
                        'hair_spray', 'academic_gown', 'dome', 'crate', 'wig', 'burrito', 'pill_bottle',
                        'chain_mail', 'theater_curtain', 'window_shade', 'barrel', 'washbasin',
                        'ballpoint', 'basketball', 'bath_towel', 'cowboy_boot', 'gown', 'window_screen',
                        'agaric', 'cellular_telephone', 'nipple', 'barbell', 'mailbox', 'lab_coat',
                        'fire_screen', 'minibus', 'packet', 'maze', 'pole', 'horizontal_bar',
                        'sombrero', 'pickelhaube', 'rain_barrel', 'wallet', 'cassette_player',
                        'comic_book', 'piggy_bank', 'street_sign', 'bell_cote', 'fountain_pen',
                        'windsor_tie', 'volleyball', 'overskirt', 'sarong', 'purse', 'bolo_tie', 'bib',
                        'parachute', 'sleeping_bag', 'television', 'swimming_trunks', 'measuring_cup',
                        'espresso', 'pizza', 'breastplate', 'shopping_basket', 'wooden_spoon',
                        'saltshaker', 'chocolate_sauce', 'ballplayer', 'goblet', 'gyromitra',
                        'stretcher', 'water_bottle', 'dial_telephone', 'soap_dispenser', 'jersey',
                        'school_bus', 'jigsaw_puzzle', 'plastic_bag', 'reflex_camera', 'diaper',
                        'band_aid', 'ice_lolly', 'velvet', 'tennis_ball', 'gasmask', 'doormat',
                        'loafer', 'ice_cream', 'pretzel', 'quilt', 'maillot', 'tape_player', 'clog',
                        'ipod', 'bolete', 'scuba_diver', 'pitcher', 'matchstick', 'bikini', 'sock',
                        'cd_player', 'lens_cap', 'thatch', 'vault', 'beaker', 'bubble', 'cheeseburger',
                        'parallel_bars', 'flagpole', 'coffee_mug', 'rubber_eraser', 'stole',
                        'carbonara', 'dumbbell')
