# %% imports
# import numpy as np
import torch 
import transformers
import argparse
from datetime import datetime

# %% define the model

# Enable Tensor Core
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

names = {
    'albert-base-v1':
        [('prajjwal1/albert-base-v1-mnli', transformers.AlbertForSequenceClassification)],
    'albert-large-v1':
        [],
    'albert-xlarge-v1':
        [],
    'albert-xxlarge-v1':
        [('replydotai/albert-xxlarge-v1-finetuned-squad2', transformers.AlbertForQuestionAnswering),
         ('armageddon/albert-squad-v2-covid-qa-deepset', transformers.AlbertForQuestionAnswering)],
    'albert-base-v2':
        [('ArBert/albert-base-v2-finetuned-ner', transformers.AlbertForTokenClassification),
         ('anirudh21/albert-base-v2-finetuned-rte', transformers.AlbertForSequenceClassification)],
    'albert-large-v2':
        [('anirudh21/albert-large-v2-finetuned-wnli', transformers.AlbertForSequenceClassification),
         ('anirudh21/albert-large-v2-finetuned-mnli', transformers.AlbertForSequenceClassification)],
    'albert-xlarge-v2': # Let's drop it.
        [('anirudh21/albert-xlarge-v2-finetuned-mrpc', transformers.AlbertForSequenceClassification)],
    'albert-xxlarge-v2':
        [('danlou/albert-xxlarge-v2-finetuned-csqa', transformers.AutoModelForMultipleChoice),# TODO. Odd task
         ('anirudh21/albert-xxlarge-v2-finetuned-wnli', transformers.AlbertForSequenceClassification),
         ('mfeb/albert-xxlarge-v2-squad2', transformers.AlbertForQuestionAnswering)],
    'roberta-base':
        [('dominiqueblok/roberta-base-finetuned-ner', transformers.RobertaForTokenClassification),
         ('jxuhf/roberta-base-finetuned-cola', transformers.RobertaForSequenceClassification),
         ('deepset/roberta-base-squad2', transformers.RobertaForQuestionAnswering),
         ('cardiffnlp/twitter-roberta-base-sentiment', transformers.RobertaForSequenceClassification)],
    'roberta-large':
        [('danlou/roberta-large-finetuned-csqa', transformers.RobertaForMultipleChoice),# TODO. Odd task
         ('phiyodr/roberta-large-finetuned-squad2', transformers.RobertaForQuestionAnswering),
         ('mrm8488/roberta-large-finetuned-wsc', transformers.RobertaForMaskedLM), # TODO. Odd task: Task identified.
         ('philschmid/roberta-large-finetuned-clinc', transformers.RobertaForSequenceClassification),
         ('deepset/roberta-large-squad2', transformers.RobertaForQuestionAnswering),
         ('roberta-large-mnli', transformers.RobertaForSequenceClassification)],
    'distilbert-base-uncased':
        [('distilbert-base-uncased-finetuned-sst-2-english', transformers.DistilBertForSequenceClassification),
         ('elastic/distilbert-base-uncased-finetuned-conll03-english', transformers.DistilBertForTokenClassification),
         ('huggingface/distilbert-base-uncased-finetuned-mnli', transformers.DistilBertForSequenceClassification),
         ('kurianbenoy/distilbert-base-uncased-finetuned-imdb', transformers.DistilBertForSequenceClassification),
         ('nishmithaur/distilbert-base-uncased-finetuned-ner', transformers.DistilBertForTokenClassification),
         ('transformersbook/distilbert-base-uncased-finetuned-emotion', transformers.DistilBertForSequenceClassification),
         ('mosesju/distilbert-base-uncased-finetuned-news', transformers.DistilBertForSequenceClassification),
         ('malduwais/distilbert-base-uncased-finetuned-ner', transformers.DistilBertForTokenClassification),
         ('huggingface-course/distilbert-base-uncased-finetuned-imdb', transformers.DistilBertForMaskedLM),# TODO. Odd task: Task identified.
         ('typeform/distilbert-base-uncased-mnli', transformers.DistilBertForSequenceClassification),
        ],
    'microsoft/deberta-base':
        [('brandon25/deberta-base-finetuned-ner', transformers.DebertaForTokenClassification),
         ('microsoft/deberta-base-mnli', transformers.DebertaForSequenceClassification),
         ('Roberta55/deberta-base-mnli-finetuned-cola', transformers.DebertaForSequenceClassification)],
    'microsoft/deberta-large':
        [('microsoft/deberta-large-mnli', transformers.DebertaForSequenceClassification),
         ('Narsil/deberta-large-mnli-zero-cls', transformers.DebertaForSequenceClassification)],
    'microsoft/deberta-xlarge': # Problematic
        [('microsoft/deberta-xlarge-mnli', transformers.DebertaForSequenceClassification)],
    'microsoft/deberta-v3-large':
        [('yangheng/deberta-v3-large-absa', transformers.DebertaV2Model), # TODO Ready
         ('SetFit/deberta-v3-large__sst2__train-16-8', transformers.DebertaV2ForSequenceClassification)], # TODO Ready
    'microsoft/deberta-v3-xsmall':
        [('philschmid/deberta-v3-xsmall-emotion', transformers.DebertaV2ForSequenceClassification), # TODO Ready
         ('nbroad/deberta-v3-xsmall-squad2', transformers.DebertaV2ForQuestionAnswering), # TODO Ready
         ('domenicrosati/deberta-mlm-test', transformers.DebertaV2ForMaskedLM)], # TODO Ready
    # 'microsoft/deberta-v2-xxlarge': # Model too large
    #     [('microsoft/deberta-v2-xxlarge-mnli', transformers.DebertaV2ForSequenceClassification)],
    'microsoft/deberta-v2-xlarge':
        [('microsoft/deberta-v2-xlarge-mnli', transformers.DebertaV2ForSequenceClassification)],# TODO Ready
    'microsoft/deberta-v3-small':
        [('mrm8488/deberta-v3-small-finetuned-cola', transformers.DebertaV2ForSequenceClassification),
         ('mrm8488/deberta-v3-small-finetuned-mnli', transformers.DebertaV2ForSequenceClassification),
         ('mrm8488/deberta-v3-small-finetuned-mrpc', transformers.DebertaV2ForSequenceClassification),
         ('mrm8488/deberta-v3-small-finetuned-sst2', transformers.DebertaV2ForSequenceClassification)],
    'facebook/bart-base':
        [('mse30/bart-base-finetuned-arxiv', transformers.BartForConditionalGeneration),
         ('VictorSanh/bart-base-finetuned-xsum', transformers.BartForConditionalGeneration),
         ('valhalla/bart-large-finetuned-squadv1', transformers.BartForQuestionAnswering),
        ],
    'facebook/bart-large':
        [('phiyodr/bart-large-finetuned-squad2', transformers.BartForQuestionAnswering),
         ('facebook/bart-large-mnli', transformers.BartForSequenceClassification),
         ('facebook/bart-large-cnn', transformers.BartForConditionalGeneration),
         ('facebook/bart-large-xsum', transformers.BartForConditionalGeneration),
         ('philschmid/bart-large-cnn-samsum', transformers.BartForConditionalGeneration)],
    't5-base':
        [('mrm8488/t5-base-finetuned-common_gen', transformers.T5ForConditionalGeneration),
         ('mrm8488/t5-base-finetuned-break_data', transformers.T5ForConditionalGeneration),
         ('mrm8488/t5-base-finetuned-squadv2', transformers.T5ForConditionalGeneration),
         ('MaRiOrOsSi/t5-base-finetuned-question-answering', transformers.T5ForConditionalGeneration),
         ('mrm8488/t5-base-finetuned-emotion', transformers.T5ForConditionalGeneration),
         ('mrm8488/t5-base-finetuned-qasc', transformers.T5ForConditionalGeneration)],
    't5-large':
        [('google/t5-large-ssm', transformers.T5ForConditionalGeneration),
         ('google/t5-large-ssm-nq', transformers.T5ForConditionalGeneration),
        ],
    't5-small':
        [('google/t5-small-ssm', transformers.T5ForConditionalGeneration),
         ('hetpandya/t5-small-tapaco', transformers.T5ForConditionalGeneration),
         ('valhalla/t5-small-qg-hl', transformers.T5ForConditionalGeneration),
         ('valhalla/t5-small-qa-qg-hl', transformers.T5ForConditionalGeneration)],
    'gpt2':
        [('mrm8488/GPT-2-finetuned-common_gen', transformers.GPT2LMHeadModel),
         ('mrm8488/GPT-2-finetuned-CORD19', transformers.GPT2LMHeadModel),
         ('lighteternal/gpt2-finetuned-greek', transformers.GPT2LMHeadModel),
         ('mrm8488/gpt2-finetuned-recipes-cooking', transformers.GPT2LMHeadModel),
         ('lighteternal/gpt2-finetuned-greek-small', transformers.GPT2LMHeadModel),
         # ('Rocketknight1/gpt2-finetuned-wikitext2', transformers.GPT2LMHeadModel), # TODO: Tensorflow
         ('BenDavis71/GPT-2-Finetuning-AIRaid', transformers.GPT2LMHeadModel)],
    'gpt2-medium':
        [('ayameRushia/gpt2-medium-fine-tuning-indonesia-poem', transformers.GPT2LMHeadModel),
         ('ml6team/gpt2-medium-dutch-finetune-oscar', transformers.GPT2LMHeadModel),
         ('ml6team/gpt2-medium-german-finetune-oscar', transformers.GPT2LMHeadModel),
         ('Pyjay/gpt2-medium-dutch-finetuned-text-generation', transformers.GPT2LMHeadModel)],
    'gpt2-large':
        [('mrm8488/GPT-2-finetuned-common_gen', transformers.GPT2LMHeadModel),
        ('aliosm/ComVE-gpt2-large', transformers.GPT2LMHeadModel),
        ],
    'distilgpt2':
        [('MYX4567/distilgpt2-finetuned-wikitext2', transformers.GPT2LMHeadModel),
         ('mahaamami/distilgpt2-finetuned-wikitext2', transformers.GPT2LMHeadModel),
         ('mrm8488/distilgpt2-finetuned-bookcopus-10', transformers.GPT2LMHeadModel),
         ('mrm8488/distilgpt2-finetuned-wsb-tweets', transformers.GPT2LMHeadModel),
         ('mgfrantz/distilgpt2-finetuned-reddit-tifu', transformers.GPT2LMHeadModel),
        ],
    'xlm-roberta-base':
        [#('airesearch/xlm-roberta-base-finetuned', transformers.XLMRobertaForMaskedLM), # Model not in the link
         ('Davlan/xlm-roberta-base-finetuned-kinyarwanda', transformers.XLMRobertaForMaskedLM),
         ('Davlan/xlm-roberta-base-finetuned-luo', transformers.XLMRobertaForMaskedLM),
         ('Davlan/xlm-roberta-base-finetuned-amharic', transformers.XLMRobertaForMaskedLM),
         ('Davlan/xlm-roberta-base-finetuned-wolof', transformers.XLMRobertaForMaskedLM),
         ('erst/xlm-roberta-base-finetuned-nace', transformers.XLMRobertaForSequenceClassification),
         ('Davlan/xlm-roberta-base-finetuned-yoruba', transformers.XLMRobertaForMaskedLM),
         ('be4rr/xlm-roberta-base-finetuned-panx-de', transformers.XLMRobertaForTokenClassification),
         ('RobertoMCA97/xlm-roberta-base-finetuned-panx-fr', transformers.XLMRobertaForTokenClassification),
         ('Davlan/xlm-roberta-base-finetuned-naija', transformers.XLMRobertaForMaskedLM),
         ('mbeukman/xlm-roberta-base-finetuned-ner-luganda', transformers.XLMRobertaForTokenClassification),
         ('Davlan/xlm-roberta-base-finetuned-igbo', transformers.XLMRobertaForMaskedLM),
         ('salesken/xlm-roberta-base-finetuned-mnli-cross-lingual-transfer', transformers.XLMRobertaForSequenceClassification),
         ('transformersbook/xlm-roberta-base-finetuned-panx-en', transformers.XLMRobertaForTokenClassification),
         ('cardiffnlp/twitter-xlm-roberta-base-sentiment', transformers.XLMRobertaForSequenceClassification),
         ('deepset/xlm-roberta-base-squad2', transformers.XLMRobertaForQuestionAnswering)],
    'xlm-roberta-large':  # Too large model
        [('xlm-roberta-large-finetuned-conll03-english', transformers.XLMRobertaForTokenClassification),
        ('sontn122/xlm-roberta-large-finetuned-squad', transformers.XLMRobertaForQuestionAnswering),
        ('xlm-roberta-large-finetuned-conll02-spanish', transformers.XLMRobertaForTokenClassification),
        ('xlm-roberta-large-finetuned-conll03-german', transformers.XLMRobertaForTokenClassification),
        ('FrGes/xlm-roberta-large-finetuned-EUJAV-datasetA', transformers.XLMRobertaForSequenceClassification),
        ('joeddav/xlm-roberta-large-xnli', transformers.XLMRobertaForSequenceClassification)],  # Too large model
    'hfl/chinese-macbert-base':
        [],
    'EleutherAI/gpt-neo-1.3B':
        [('groar/gpt-neo-1.3B-finetuned-escape2', transformers.GPTNeoForCausalLM),
         ('KoboldAI/GPT-Neo-1.3B-Adventure', transformers.GPTNeoForCausalLM)],
    'EleutherAI/gpt-neo-125M':
        [('b3ck1/gpt-neo-125M-finetuned-beer-recipes', transformers.GPTNeoForCausalLM),
        ],
    'EleutherAI/gpt-neo-2.7B':
        [('KoboldAI/GPT-Neo-2.7B-Picard', transformers.GPTNeoForCausalLM)
        ],
    'xlnet-base-cased':
        [('anirudh21/xlnet-base-cased-finetuned-rte', transformers.XLNetForSequenceClassification),
        ],
    'xlnet-large-cased':
        [],
    'SpanBERT/spanbert-base-cased':
        ['mrm8488/spanbert-base-finetuned-tacred', #Auto is fine
         ('mrm8488/spanbert-base-finetuned-squadv1', transformers.BertForQuestionAnswering),
         ('mrm8488/spanbert-base-finetuned-squadv2', transformers.BertForQuestionAnswering),
         ('anas-awadalla/spanbert-base-cased-few-shot-k-32-finetuned-squad-seed-4', transformers.BertForQuestionAnswering),
        ],
    'bert-base-multilingual-cased':
        [('henryk/bert-base-multilingual-cased-finetuned-polish-squad2', transformers.BertForQuestionAnswering),
         ('salti/bert-base-multilingual-cased-finetuned-squad', transformers.BertForQuestionAnswering),
         ('henryk/bert-base-multilingual-cased-finetuned-dutch-squad2', transformers.BertForQuestionAnswering),
         ('Davlan/bert-base-multilingual-cased-finetuned-amharic', transformers.BertForMaskedLM),
         ('mrm8488/bert-multi-cased-finetuned-xquadv1', transformers.BertForQuestionAnswering),
         ('Davlan/bert-base-multilingual-cased-finetuned-yoruba', transformers.BertForMaskedLM)],
    'DeepPavlov/rubert-base-cased':
        [('DeepPavlov/rubert-base-cased-conversational', transformers.BertForMaskedLM), # Not perfect, but close
         ('chrommium/rubert-base-cased-sentence-finetuned-headlines_X', transformers.BertForSequenceClassification),
        ],
    'bert-base-uncased':
        [('victoraavila/bert-base-uncased-finetuned-squad', transformers.BertForQuestionAnswering),
         ('ikevin98/bert-base-uncased-finetuned-sst2', transformers.BertForSequenceClassification),
         ('Jorgeutd/bert-base-uncased-finetuned-surveyclassification', transformers.BertForSequenceClassification),
         ('ncduy/bert-base-uncased-finetuned-swag', transformers.BertForMultipleChoice),  # RuntimeError
         ('kaporter/bert-base-uncased-finetuned-squad', transformers.BertForQuestionAnswering),
         ('avb/bert-base-uncased-finetuned-cola', transformers.BertForSequenceClassification),
         ('anirudh21/bert-base-uncased-finetuned-qnli', transformers.BertForSequenceClassification),
         ('ikevin98/bert-base-uncased-finetuned-sst2-sst2-membership', transformers.BertForSequenceClassification),  # Task unknown
         ('gchhablani/bert-base-cased-finetuned-sst2', transformers.BertForSequenceClassification),
         ('dslim/bert-base-NER-uncased', transformers.BertForTokenClassification),
         ('echarlaix/bert-base-uncased-sst2-acc91.1-d37-hybrid', transformers.BertForSequenceClassification)
        ],
    'bert-large-uncased':
        [('bert-large-uncased-whole-word-masking-finetuned-squad', transformers.BertForQuestionAnswering),
         ('Jorgeutd/bert-large-uncased-finetuned-ner', transformers.BertForTokenClassification),
         ('SauravMaheshkar/clr-finetuned-bert-large-uncased', transformers.BertForMaskedLM), # It's probably Roberta
         ('tiennvcs/bert-large-uncased-finetuned-infovqa', transformers.BertForQuestionAnswering),
         ('lewtun/bert-large-uncased-wwm-finetuned-boolq', transformers.BertForSequenceClassification),
         ('tiennvcs/bert-large-uncased-finetuned-infovqa', transformers.BertForQuestionAnswering),
         ('echarlaix/bert-large-uncased-whole-word-masking-finetuned-sst-2', transformers.BertForSequenceClassification),
         ('YeRyeongLee/bert-large-uncased-finetuned-filtered-0602', transformers.BertForSequenceClassification)
        ],
    'camembert-base':
        [('louisdeco/camembert-base-finetuned-RankLineCause', transformers.CamembertForSequenceClassification),
         ('waboucay/camembert-base-finetuned-repnum_wl_3_classes', transformers.CamembertForSequenceClassification),
         ('waboucay/camembert-base-finetuned-xnli_fr-finetuned-nli-repnum_wl-rua_wl', transformers.CamembertForSequenceClassification),
         ('fmikaelian/camembert-base-fquad', transformers.CamembertForQuestionAnswering),
         ('louisdeco/camembert-base-finetuned-RankLineCause', transformers.CamembertForSequenceClassification),
         ('mrm8488/camembert-base-finetuned-movie-review-sentiment-analysis', transformers.CamembertForSequenceClassification)],
    'prajjwal1/bert-tiny':
        [('prajjwal1/bert-tiny-mnli', transformers.BertForSequenceClassification),
         # ('prajjwal1/bert-tiny', transformers.BertForPreTraining), # Just adding the pretrain class
        ],
    'prajjwal1/bert-small':
        [('prajjwal1/bert-small-mnli', transformers.BertForSequenceClassification)],
    'prajjwal1/bert-medium':
        [('prajjwal1/bert-medium-mnli', transformers.BertForSequenceClassification),
         # ('prajjwal1/bert-medium', transformers.BertForPreTraining)#Adding prerain task
         ],
    'prajjwal1/bert-mini':
        [('prajjwal1/bert-mini-mnli', transformers.BertForSequenceClassification),
         # ('prajjwal1/bert-mini', transformers.BertForPreTraining)#Adding prerain task
         ],
    'google/mobilebert-uncased':
        [('mrm8488/mobilebert-uncased-finetuned-squadv2', transformers.MobileBertForQuestionAnswering),
         ('mrm8488/mobilebert-uncased-finetuned-squadv1', transformers.MobileBertForQuestionAnswering),
         ('Gozdi/mobilebert-finetuned-coqa', transformers.MobileBertForQuestionAnswering), # Tensorflow checkpoint
        ],
}

sentence = ['This is an example sentence']
sentences = ["This is an example sentence",
             "A whole new sentence it is and it is longer than all the other four sentences in this script file",
             "This one actually has a period.",
             "Adding a tipo to make it interesting.",
             "This is the last sentence.",
             "Call me Ishmael.",
             "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife",
             "All happy families are alike; each unhappy family is unhappy in its own way.",
             "It was the best of times, it was the worst of times",
             "I have run out of new lines.",
             "Sometimes this place gets kind of empty.",
             "I was a bayman like my father was before",
             "Winning is a habit.",
             "Soon after that, he went home.",
             "I have seen things you people wouldn't believe."]

def run_all():
    for name in names.keys():
        for i in names[name]:
            config = transformers.AutoConfig.from_pretrained(i)
            tokenizer = transformers.AutoTokenizer.from_pretrained(name)
            model = transformers.AutoModel.from_pretrained(i)
            encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            _ = model(**encoded_input)


def run_one(pre_train_name, finetune_name=None, export_weight=False):
    if finetune_name is None:
        finetune_name = pre_train_name
    # elif finetune_name not in names[pre_train_name]:
    #     raise ValueError(f"Pretrained ({pre_train_name}) and Finetuned ({finetune_name}) models mismatch.")


    print(f"Pretrained: {pre_train_name}\nFinetuned: {finetune_name}")
    print(f"Sentence: {sentences[args.sentence - 1]}")

    # config = transformers.AutoConfig.from_pretrained(finetune_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(pre_train_name)
    model = GetFinetunedModel(finetune_name).from_pretrained(finetune_name)
    encoded_input = tokenizer(sentences[args.sentence - 1], return_tensors='pt')

    # model.half()

    if export_weight:
        count = 0
        for name, param in model.named_parameters():
            print("name: " + name)
            print("param size: ")
            print(param.data.size())
            print("param: ")
            print(param.data)

            count += 1
            weightFile = f"Weights/{finetune_name.replace('/', '_')}/{count}#{name}.pt"
            import os
            os.makedirs(os.path.dirname(weightFile), exist_ok=True)
            torch.save(param.data, weightFile)
        exit()


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Running on CPU.")

    print("Before GPU copy:", datetime.utcnow())

    encoded_input = encoded_input.to(device)
    model = model.to(device)

    print("Before Inference:", datetime.utcnow())

    if "t5" not in pre_train_name:
        _ = model(**encoded_input)
    else:
        _ = model(input_ids=encoded_input.input_ids, attention_mask=None, decoder_input_ids=encoded_input.input_ids)

    print("After Inference:", datetime.utcnow())
    print("Finished.")


def GetFinetunedModel(finetunedName: str):
    for _, finetunedModels in names.items():
        for finetunedModelVsClass in finetunedModels:
            if type(finetunedModelVsClass) is tuple:
                name, modelClass = finetunedModelVsClass
                if name == finetunedName:
                    return modelClass

    print("Could not find specific model class.")
    return transformers.AutoModelForPreTraining


def CreateBashCommand():
    with open("profile_finetuned_five(2).sh", "w") as bashFile:
        import model_pool
        for pretrainedModel in model_pool.names.keys():
            for finetunedModel in model_pool.names[pretrainedModel]:
                for i in range(1, 6):
                    bashFile.write("/usr/local/cuda/bin/ncu --print-units base --section SpeedOfLight "
                                   "--metrics dram__bytes_read,dram__bytes_write "
                                   "/home/rafi/Environments/huggingface_transformers/bin/python "
                                   "/home/rafi/huggingface_transformers/model_pool_finetuned.py "
                                   f"--pre_train_name={pretrainedModel} "
                                   f"--finetune_name={finetunedModel} "
                                   f"--sentence={i} > \"Profiles_Finetuned_5(2)/{finetunedModel.replace('/', '_')}_{i}.txt\"\n")
                bashFile.write("rm -r ~/.cache/huggingface/transformers\n\n")



# %% main
if __name__ == '__main__':
    # CreateBashCommand()
    parser = argparse.ArgumentParser(description='Process the inputs')
    parser.add_argument('--pre_train_name', type=str, help='The name of the pretrained model', required=True)
    parser.add_argument('--finetune_name', type=str, help='The name of the finetuned model', default=None)
    parser.add_argument("--sentence", type=int, default=1, choices=range(1, 16))

    args = parser.parse_args()
    run_one(args.pre_train_name, args.finetune_name, export_weight=False)