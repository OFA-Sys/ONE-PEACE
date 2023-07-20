import torch
import cv2
from one_peace.models import from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
	"ONE-PEACE_Grounding",
    model_type="one_peace_classify",
    device=device,
    dtype="float32"
)

# process raw data
image_text_list = [
    ("assets/pokemons.jpg", "a blue turtle-like pokemon with round head"),
    ("assets/pokemons.jpg", "Bulbasaur"),
    ("assets/pokemons.jpg", "Charmander"),
    ("assets/pokemons.jpg", "Squirtle"),
    ("assets/one_piece.jpeg", "Brook"),
    ("assets/one_piece.jpeg", "Franky"),
    ("assets/one_piece.jpeg", "Monkey D. Luffy"),
    ("assets/one_piece.jpeg", "Nami"),
    ("assets/one_piece.jpeg", "Nico Robin"),
    ("assets/one_piece.jpeg", "Roronoa Zoro"),
    ("assets/one_piece.jpeg", "Tony Tony Chopper"),
    ("assets/one_piece.jpeg", "Usopp"),
    ("assets/one_piece.jpeg", "Vinsmoke Sanji"),
]
(src_images, image_widths, image_heights), src_tokens  = model.process_image_text_pairs(
    image_text_list, return_image_sizes=True
)

with torch.no_grad():
    # extract features
    vl_features = model.extract_vl_features(src_images, src_tokens).sigmoid()
    # extract coords
    vl_features[:, ::2] *= image_widths.unsqueeze(1)
    vl_features[:, 1::2] *= image_heights.unsqueeze(1)
    coords = vl_features.cpu().tolist()

# display results
for i, image_text_pair in enumerate(image_text_list):
    image, text = image_text_pair
    img = cv2.imread(image)
    cv2.rectangle(
        img,
        (int(coords[i][0]), int(coords[i][1])),
        (int(coords[i][2]), int(coords[i][3])),
        (0, 255, 0),
        3
    )
    print(image, text)
    cv2.imshow('visual_grounding', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
