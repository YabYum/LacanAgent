import random
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def draw_stickman(ax, head_coords, limb_lengths, angles):
    """
    Draw a stickman on the given axis.

    Parameters:
    - ax: Matplotlib axis to draw on.
    - head_coords: Coordinates of the head (x, y).
    - limb_lengths: Dictionary with lengths of limbs (arms and legs).
    - angles: Dictionary with angles (in degrees) of limbs.
    """

    # Calculate limb positions based on angles and lengths
    def get_limb_coords(start, length, angle):
        return (start[0] + length * np.cos(np.radians(angle)),
                start[1] + length * np.sin(np.radians(angle)))

    # Draw the head
    head_radius = 0.5
    head_circle = plt.Circle(head_coords, head_radius, fill=True, color='black')
    ax.add_artist(head_circle)

    # Body (from head center to body center)
    body_length = limb_lengths['body']
    body_bottom = (head_coords[0], head_coords[1] - (head_radius + body_length))
    ax.plot([head_coords[0], body_bottom[0]], [head_coords[1], body_bottom[1]], 'k-')

    # Arms
    left_arm_angle = angles['left_arm']
    right_arm_angle = angles['right_arm']
    left_arm_end = get_limb_coords(body_bottom, limb_lengths['arm'], left_arm_angle + 90)
    right_arm_end = get_limb_coords(body_bottom, limb_lengths['arm'], right_arm_angle - 90)
    ax.plot([body_bottom[0], left_arm_end[0]], [body_bottom[1] + 1.5, left_arm_end[1]], 'k-')
    ax.plot([body_bottom[0], right_arm_end[0]], [body_bottom[1] + 1.5, right_arm_end[1]], 'k-')

    # Legs
    left_leg_angle = angles['left_leg']
    right_leg_angle = angles['right_leg']
    left_leg_end = get_limb_coords(body_bottom, limb_lengths['leg'], left_leg_angle + 90)
    right_leg_end = get_limb_coords(body_bottom, limb_lengths['leg'], right_leg_angle - 90)
    ax.plot([body_bottom[0], left_leg_end[0]], [body_bottom[1], left_leg_end[1]], 'k-')
    ax.plot([body_bottom[0], right_leg_end[0]], [body_bottom[1], right_leg_end[1]], 'k-')


def generate_random_angles():
    return {
        'left_arm': np.random.uniform(30, 145),
        'right_arm': np.random.uniform(30, 145),
        'left_leg': np.random.uniform(100, 175),
        'right_leg': np.random.uniform(5, 70)
    }


# Generate a random stickman
def generate_image(name):

    fig, ax = plt.subplots()
    limb_lengths = {'arm': 1.5, 'leg': 2.5, 'body': 3}
    head_coords = (5, 10)
    angles = generate_random_angles()

    # Set limits and aspect
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal')

    # Draw the stickman
    draw_stickman(ax, head_coords, limb_lengths, angles)

    # Hide the axes
    plt.axis('off')
    plt.show()


# Load and preprocess images
def load_images(image_paths):
    images = []
    for path in image_paths:
        # Load the image
        img = load_img(path, color_mode='grayscale', target_size=(64, 64))  # adjust target_size accordingly
        # Convert the image to an array
        img_array = img_to_array(img)
        # Normalize the image
        img_array = img_array / 255.0
        images.append(img_array)
    return np.array(images)


def w2v():
    corpus = [['Environmental Engineer'], ['Advocate for Kindness'], ['Chess Enthusiast'],['Devoted Mother'],['Optimistic Dreamer'],['World Traveler'],['Aspiring Artist'],['Cultural Ambassador'],['Animal Rights Activist'],['Tech Innovator']]

    w2vid = Word2Vec(corpus, vector_size=4, window=2, min_count=1, workers=1)
    word_vectors = w2vid.wv.vectors
    scaler = MinMaxScaler(feature_range=(0, 5))
    normalized_vectors = scaler.fit_transform(word_vectors)
    for i, word in enumerate(w2vid.wv.index_to_key):
        w2vid.wv[word] = normalized_vectors[i]

    word_vectors = w2vid.wv.vectors
    words = w2vid.wv.index_to_key

    return w2vid


class CharacterDataset(Dataset):
    def __init__(self, input_vectors, output_vectors):
        self.input_vectors = torch.tensor(input_vectors, dtype=torch.float32)
        self.output_vectors = torch.tensor(output_vectors, dtype=torch.float32)

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        return self.input_vectors[idx], self.output_vectors[idx]


def load_data():
    cnn = load_model('cnn.h5')
    id2vec = w2v()
    with open('10characters.json') as file:
        data = json.load(file)
    names = [character['name'] for character in data['characters']]
    
    def get_input_output(name):
        image_path = f'character_images/{name}.png'
        image = load_images([image_path])
        vison = np.array(cnn(image))
        for character in data['characters']:
            if character['name'] == name:
                identity = character['identity']
                break
        vc = id2vec.wv[identity]
        return vc, vison[0]
    
    input_vectors = []
    output_vectors = []
    
    for name in names:
        vector, vision = get_input_output(name)
        input_vectors.append(vision)
        output_vectors.append(vector)
    
    return input_vectors, output_vectors, id2vec, cnn


def create_character_dataloader(batch_size=4, shuffle=True):
    input_vectors, output_vectors, id2vec, cnn = load_data()
    dataset = CharacterDataset(input_vectors, output_vectors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, id2vec, cnn
