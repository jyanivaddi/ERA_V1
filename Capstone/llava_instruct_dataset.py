import requests
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Sequence


def split_data_to_train_and_val(data_path):
    list_data_dict = json.load(open(data_path, "r"))
    num_samples = len(list_data_dict)
    rand_indices = np.arange(num_samples)
    np.random.shuffle(rand_indices)
    val_indices = rand_indices[:100]
    train_indices = rand_indices[100:]
    train_data = [list_data_dict[i] for i in train_indices]
    val_data = [list_data_dict[i] for i in val_indices]
    return train_data, val_data


def get_image_embeddings(image_url, model, preprocessor, device=None):
    """
    This method computes the clip embeddings for a given image, after preprocessing it according to the model
    """
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except:
        return None
    processed_image = preprocessor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**processed_image)
    return outputs.last_hidden_state.squeeze()[1:,:]



class LlavaFinetuneDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 list_data_dict,
                 tokenizer,
                 clip_model,
                 clip_preprocessor,
                 device,
                 max_seq_len=64):
        super(LlavaFinetuneDataset, self).__init__()
        self.list_data_dict = list_data_dict
        self.tokenizer = tokenizer
        self.clip_model = clip_model
        self.clip_preprocessor = clip_preprocessor
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = 50256
        self.COMMENT_TOKEN_ID = 23893
        self.IMAGE_TOKEN_ID = 5159
        self.IMAGE_START_TOKEN = "<image>"
        self.seq_len = seq_len
        self.image_embedding_size = 49 
        #self.max_question_len = self.max_seq_len - (1+self.image_embedding_size+2) # 64 - (<image token>, 49 , <comment>, eos)
        self.max_question_len = 32


    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        {'id': '000000033471',
        'image': '000000033471.jpg',
        'conversations': [{'from': 'human',
        'value': '<image>\nWhat are the colors of the bus in the image?'},
        {'from': 'gpt', 'value': 'The bus in the image is white and red.'},
        {'from': 'human',
        'value': 'What feature can be seen on the back of the bus?'},
        {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'},
        {'from': 'human',
        'value': 'Is the bus driving down the street or pulled off to the side?'},
        {'from': 'gpt',
        'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}
        """
        image_file = self.list_data_dict[i]['image']
        image_url = f"http://images.cocodataset.org/train2017/{image_file}"
        image_embeddings = get_image_embeddings(image_url, self.clip_model, self.clip_preprocessor)
        if image_embeddings is None:
            print("warning: skipping due to invalid image path")
            return None
        conversation = self.extract_conversation(self.list_data_dict[i]['conversations'])
        conversation['image_embeddings'] = image_embeddings
        return conversation


    def tokenize_sentence(self, sentence, filter_long_sentence = True):
        tokenizer_output = self.tokenizer(sentence, return_tensors="pt", return_attention_mask=False)
        tokenized_sentence = tokenizer_output['input_ids'].squeeze()
        return tokenized_sentence


    def extract_conversation(self, conversations_list):
        """
        if there are multiple conversations, lets pick one of them
        """
        assert len(conversations_list) % 2 == 0 # we always want to make sure there are pairs of conversations
        num_conversations = int(conversations_list/2)
        # lets pick a random conversational index
        tgt_idx = np.random.randint(0, num_conversations)  
        ques_idx = 2*tgt_idx-2
        ans_idx = ques_idx + 1
        assert conversations_list[ques_idx]['from'] == 'human'
        assert conversations_list[ans_idx]['from'] == 'gpt'
        ques = conversations_list[ques_idx]['value'].strip("<image>\n")
        ans = conversations_list[ans_idx]['value']
        ques_tokenized = self.tokenize_sentence(ques, filter_long_sentence = True)
        ans_tokenized = self.tokenize_sentence(ans, filter_long_sentence = False)
        conversation = {}
        conversation['ques_tokenized'] = ques_tokenized
        conversation['ans_tokenized'] = ans_tokenized
        conversation['question'] = ques
        conversation['answer'] = ans
        return conversation


@dataclass
class LlavaCollator(object):
    """
    Collate examples for Llava fine-tuning dataset
    if any of the value is None
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        for instance in instances:
            if any(value is None for value in instance.values()):
                return None
        image_embeddings, ques_tokenized, ans_tokenized, questions, answers = tuple([instance[key] for instance in instances]
                                  for key in ("image_embeddings", "ques_tokenized", "ans_tokenized", "question","answer"))
        ques_tokenized = torch.nn.utils.rnn.pad_sequence(
            ques_tokenized,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id)
        ans_tokenized = torch.nn.utils.rnn.pad_sequence(ans_tokenized,
                                                 batch_first=True,
                                                 padding_value=self.tokenizer.eos_token_id)
        ques_tokenized = ques_tokenized[:, :max_length]
        ans_tokenized = ans_tokenized[:, :max_length]
        batch = dict(
            ques_tokenized=ques_tokenized,
            ans_tokenized=ans_tokenized,
            attention_mask=ques_tokenized.ne(self.tokenizer.pad_token_id),
        )

        batch['image_embeddings'] = torch.stack(image_embeddings)

        return batch


