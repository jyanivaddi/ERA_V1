import requests
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Sequence





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
                 data_path,
                 tokenizer,
                 clip_model,
                 clip_preprocessor,
                 device,
                 max_seq_len=64):
        super(LlavaFinetuneDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
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
        self.max_question_len = self.max_seq_len - (1+self.image_embedding_size+2) # 64 - (<image token>, 49 , <comment>, eos)


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
        if not image_embeddings:
            return None
        conversation = self.extract_conversation(self.list_data_dict[i]['conversations'])
        conversation['image_embeddings'] = image_embeddings
        return conversation


    def tokenize_caption(self, sentence, crop_if_needed = True):
        tokenizer_output = self.tokenizer(sentence, return_tensors="pt", return_attention_mask=False)
        tokenized_sentence = tokenizer_output['input_ids'].squeeze()
        if crop_if_needed:
            if len(tokenized_sentence) > self.max_question_len: # token is longer than max lengh, crop and add eos token
                tokenized_sentence = torch.cat([tokenized_caption[:self.max_caption_len], torch.tensor([self.eos_token_id], dtype=torch.int64)], dim=0)
            else: # token is shorter than max length - pad and add eos token
                num_padding_tokens = self.max_caption_len - len(tokenized_caption) + 1
                tokenized_caption = torch.cat([tokenized_caption, torch.tensor([self.eos_token_id] * num_padding_tokens, dtype=torch.int64)], dim=0)
        return tokenized_caption


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
        ques_tokenized = self.tokenize_sentence(ques, crop_if_needed = True)
        ans_tokenized = self.tokenize_sentence(ans, crop_if_needed = False)
        return


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
