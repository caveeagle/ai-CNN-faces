
from facenet_pytorch import InceptionResnetV1
import torch

model = InceptionResnetV1(pretrained='vggface2').eval()
print("OK")

