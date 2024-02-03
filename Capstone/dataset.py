import requests
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def fetch_captions_and_images(json_path):
    """
    Read a JSON file and return its contents as a dictionary.

    Parameters:
    - file_path (str): The path to the JSON file.

    Returns:
    - dict: The contents of the JSON file as a dictionary.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    captions = {}
    image_paths = {}
    image_ids = []
    annotations = data['annotations']
    images = data['images']

    for img in images:
        image_paths[img['id']] = img['coco_url']
        image_ids.append(img['id'])
    
    for annotation in annotations:
        captions[annotation['image_id']] = annotation['caption']
    
    print(f"total image ids: {len(image_paths)}, total images: {len(image_paths)}, total captions: {len(captions)}")
    return captions, image_paths, image_ids


def get_image_embeddings(image_url, model, preprocessor, device=None):
    """
    This method computes the clip embeddings for a given image, after preprocessing it according to the model
    """
    image = Image.open(requests.get(image_url, stream=True).raw)

    processed_image = preprocessor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**processed_image)
    return outputs.last_hidden_state.squeeze()[1:,:]

class PreTrainDataset(Dataset):

    def __init__(self, 
                 image_ids,
                 raw_images_list,
                 captions,
                 tokenizer, 
                 clip_model,
                 clip_preprocessor,
                 device,
                 seq_len=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.ds = None
        self.clip_model = clip_model
        self.clip_preprocessor = clip_preprocessor
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = 50256
        self.COMMENT_TOKEN_ID = 23893
        self.IMAGE_TOKEN_ID = 5159
        self.seq_len = seq_len
        self.captions = captions
        self.raw_images_list = raw_images_list
        self.image_ids = image_ids
        self.image_embedding_size = 49 
        self.max_caption_len = self.seq_len - (1+self.image_embedding_size+2) # 64 - (<image token>, 49 , <comment>, eos)
        

    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):

        # get image embeddings
        this_img_id = "{:012d}".format(int(self.image_ids[idx]))
        image_embeddings = get_image_embeddings(self.raw_images_list[int(self.image_ids[idx])], self.clip_model, self.clip_preprocessor)
        
        # get caption
        caption = self.captions[int(this_img_id)]
        tokenized_caption = self.tokenize_caption(caption)
        
        return {
            "image_embeddings": image_embeddings,
            "caption": caption,
            "tokenized_caption": tokenized_caption,
            "token_len": len(tokenized_caption)
        }
    
    def tokenize_caption(self, caption):
        tokenizer_output = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
        tokenized_caption = tokenizer_output['input_ids'].squeeze()
        if len(tokenized_caption) > self.max_caption_len: # token is longer than max lengh, crop and add eos token
            tokenized_caption = torch.cat([tokenized_caption[:self.max_caption_len], torch.tensor([self.eos_token_id], dtype=torch.int64)], dim=0)
        else: # token is shorter than max length - pad and add eos token
            num_padding_tokens = self.max_caption_len - len(tokenized_caption) + 1
            tokenized_caption = torch.cat([tokenized_caption, torch.tensor([self.eos_token_id] * num_padding_tokens, dtype=torch.int64)], dim=0)
        return tokenized_caption


    def collate_samples(self, batch):
        """
        Perform dynamic batching on the sequences.
        For each batch, we get the length of the longest sentence and pad the remaining sentences according to that.
        """

        #print("inside collate function")
        # max encoder str length
        max_len = max(x["token_len"] for x in batch)
        #print(f"longest token in this batch: {max_len}")
        #tokenized_captions = torch.nn.utils.rnn.pad_sequence(batch['tokenized_captions'])
        #image_embeddings = torch.nn.utils.rnn.pad_sequence(batch['image_embeddings'])

        captions_list = []
        image_embeddings_list = []
        tokenized_captions_list = []

        for cnt, x in enumerate(batch):
            # Add sos, eos and padding to each sentence
            num_padding_tokens = max(0, max_len - len(x["tokenized_caption"]))+1  # we will add <s> and </s>
            image_tokens = torch.tensor([self.IMAGE_TOKEN_ID]*self.image_embedding_size,dtype=torch.int64)

            # Add <s> and </s> token
            batch_x = torch.cat(
                [
                    image_tokens,
                    x['tokenized_caption'],
                    torch.tensor([self.eos_token_id] * num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )
            #print(batch_x)
            tokenized_captions_list.append(batch_x)
            image_embeddings_list.append(x['image_embeddings'])
            captions_list.append(x['caption'])

        #print("inside get item and I am returning the dict list!")
        return {
            "image_embeddings": torch.vstack(image_embeddings_list),
            "tokenized_captions": torch.vstack(tokenized_captions_list),
            "captions": captions_list,
        }
