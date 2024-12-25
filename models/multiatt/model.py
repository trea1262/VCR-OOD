from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import masked_softmax,  replace_masked_values
from allennlp.nn import InitializerApplicator
from models.multiatt.Graph_transformer_soft_mask import EncoderLayer
import torch.nn as nn
import numpy as np

@Model.register("SGTEHG")

class SGTEHG_Model(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool = True,
                 reasoning_use_obj: bool = True,
                 reasoning_use_answer: bool = True,
                 reasoning_use_question: bool = True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(SGTEHG_Model, self).__init__(vocab)

        #self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.span_encoder = TimeDistributed(span_encoder)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
        )
        
        self.boxes_fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512 +5, 512),
            torch.nn.ReLU(inplace=True),
        )
        
        self.qa_input_dropout = nn.Dropout(0.1)
        self.qa_encoder1 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "QA1")
        self.qa_encoder2 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "QA2")
        self.qa_encoder3 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "QA3")
        #self.qa_final_ln = nn.LayerNorm(512)
        

        self.va_input_dropout = nn.Dropout(0.1)
        self.va_encoder1 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "VA1")
        self.va_encoder2 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "VA2")
        self.va_encoder3 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8, "VA3")
        #self.va_final_ln = nn.LayerNorm(512)

        self.reasoning_input_dropout = nn.Dropout(0.1)
        self.reasoning_encoder1 = EncoderLayer(512*3, 512*3, 512*3, 0.1, 0.1, 8, "R1")
        self.reasoning_encoder2 = EncoderLayer(512*3, 512*3, 512*3, 0.1, 0.1, 8, "R2") 
        #self.reasoning_final_ln = nn.LayerNorm(512*3)
        
        ##=======================useless
        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        ##========================
        
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(512*3, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 512),
        )
        
        self.final_mlp0 = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(512*3, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        
        self.score_fc = nn.Linear(1536, 1536)

        self._accuracy = CategoricalAccuracy()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.b_cls_loss = torch.nn.BCELoss()

        self.att_dict = {}

        initial_path = '/data/zoujy/ARC/CCN_kd/saves/1/features_array_test2.npy'
        initial_answer_feat = np.load(initial_path)
        self.adict = torch.nn.Parameter(torch.FloatTensor(initial_answer_feat))

        self.Wd = torch.nn.Linear(512,512)
        self.Wall = torch.nn.Linear(512,1)

        self.g0 = torch.nn.Linear(512,1)
        #self.g = torch.nn.Linear(512,1)

        initializer(self)

    def a(self,a_dict):
        a_dict = self.g0(a_dict).squeeze(2)#1000,512#
        return a_dict

    def through_dict(self, output_feat, a_dict): #b x 4 x 1/512 -> b 4 1/512
        batch_size = output_feat.size(0)#4
        q_size = output_feat.size(1)#4
        dict_size = a_dict.size(0)#1000
        dict_feat = a_dict.cuda()
        #print(output_feat.shape)
        #print(a_dict.shape)

        q_emb = output_feat.view(batch_size * q_size, -1, 512)#16,1,512
        d_emb = self.Wd(dict_feat).view(-1, dict_size, 512)#1,1000,512
        #print(q_emb.shape)
        #print(d_emb.shape)
        all_score = self.Wall(
                torch.tanh(d_emb.repeat(batch_size * q_size, 1, 1) + q_emb.repeat(1, dict_size, 1))
            
        ).view(batch_size * q_size, -1)#16,1000
        #print(all_score.shape)
        dict_final_feat = torch.bmm(
            torch.softmax(all_score, dim = -1 )
                .view(batch_size* q_size,1,-1),d_emb.view(-1, dict_size, 512).repeat(batch_size* q_size, 1, 1))
        #print(dict_final_feat.shape)16,1,512
        #dict_final_feat = self.g(dict_final_feat)
        return dict_final_feat.view(batch_size,q_size,512)

        #initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):

        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)  # bs,4,q_length(13),512+768;
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def forward(self,
                det_features:torch.Tensor,
                boxes: torch.Tensor,
                features_2d:torch.Tensor,
                box_mask: torch.LongTensor,
                det_featuresr:torch.Tensor,#
                boxesr: torch.Tensor,
                features_2dr:torch.Tensor,
                box_maskr: torch.LongTensor,#
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        det_features = det_features[:,:max_len,:]
        features_2d = features_2d[:, :max_len]

        obj_reps = det_features
        obj_reps = self.obj_downsample(obj_reps)

        #obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        
        obj_reps = torch.cat([obj_reps, features_2d], dim=-1)
        obj_reps = self.boxes_fc(obj_reps)

        # Now get the question representations
        q_rep_ori, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps)  #
        a_rep_ori, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps)  #

        o_rep = obj_reps ## [b,max_len,512]

        ## [b,4,len]
        max_answer_len = int(answer_mask.sum(dim=2).max().item())
        max_question_len = int(question_mask.sum(dim=2).max().item())

        q_rep = q_rep_ori[:,:,:max_question_len,:]
        a_rep = a_rep_ori[:,:,:max_answer_len,:]
        question_mask = question_mask[:,:,:max_question_len]
        answer_mask = answer_mask[:,:,:max_answer_len]

        ## construct VA graph
        va_v_rep = o_rep.unsqueeze(1).repeat(1,4,1,1)
        va_v_rep = va_v_rep.view(va_v_rep.shape[0] * va_v_rep.shape[1], va_v_rep.shape[2], va_v_rep.shape[3])
        va_a_rep = a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3])
        va_nodes = torch.cat([va_v_rep, va_a_rep],-2) ## [b*4,o_l+a_l, 512]
        
        va_v_mask = box_mask.unsqueeze(1).repeat(1,4,1)
        va_v_mask = va_v_mask.view(va_v_mask.shape[0] * va_v_mask.shape[1], va_v_mask.shape[2], 1)
        va_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        va_mask = torch.cat([va_v_mask, va_a_mask],-2)

        va_nodes1 = self.va_input_dropout(va_nodes)

        ## va encoder
        va_output1, embedding_V1 ,va_A1 = self.va_encoder1(va_nodes1, max_len, va_mask, features_2d[:,:,:2])

        self.att_dict["va_1"] = embedding_V1

        masked_va_output1 = va_output1*va_mask
        va_nodes2 = masked_va_output1 #va_nodes1 + masked_va_output1
        #va_nodes2 = va_nodes1 + masked_va_output1
        #va_A1 = 
        va_output2, embedding_V2 , va_A2 = self.va_encoder2(va_nodes2, max_len, va_mask, features_2d[:,:,:2], att_bias=va_A1)

        self.att_dict["va_2"] = embedding_V2

        masked_va_output2 = va_output2*va_mask 
        va_nodes3 = masked_va_output2 #va_nodes2 + masked_va_output2
        #va_nodes3 = va_nodes2 + masked_va_output2
        va_output3, embedding_V3 , va_A3 = self.va_encoder3(va_nodes3, max_len, va_mask, features_2d[:,:,:2], att_bias=va_A2)

        self.att_dict["va_3"] = embedding_V3

        va_output = va_output3*va_mask
        #va_output = self.va_final_ln(va_output)
        va_output = va_output[:,o_rep.shape[1]:,:].view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], 512)
       
        ## construct QA graph
        qa_q_rep = q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3])
        qa_a_rep = a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3])
        qa_nodes = torch.cat([qa_q_rep, qa_a_rep],-2) ## [b*4,q_l+a_l, 512]
        
        qa_q_mask = question_mask.view(question_mask.shape[0] * question_mask.shape[1], question_mask.shape[2], 1)
        qa_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        qa_mask = torch.cat([qa_q_mask, qa_a_mask],-2)
        
        qa_nodes1 = self.qa_input_dropout(qa_nodes)

        ## qa encoder
        qa_output1, embedding_Q1 , qa_A1 = self.qa_encoder1(qa_nodes1, max_question_len, qa_mask)
        # print("qa_A1",qa_A1[0,0,0:16,0:16])
        # print("qa_A1",qa_A1[0,1,0:16,0:16])

        self.att_dict["qa_1"] = embedding_Q1

        masked_qa_output1 = qa_output1*qa_mask
        qa_nodes2 = masked_qa_output1 #qa_nodes1 + masked_qa_output1
        #qa_nodes2 = qa_nodes1 + masked_qa_output1
        #va_A1 = 
        qa_output2, embedding_Q2 , qa_A2 = self.qa_encoder2(qa_nodes2, max_question_len, qa_mask, att_bias=qa_A1)
        # print("qa_A2",qa_A2[0,0,0:16,0:16])
        # print("qa_A2",qa_A2[0,1,0:16,0:16])

        self.att_dict["qa_2"] = embedding_Q2

        masked_qa_output2 = qa_output2*qa_mask  
        qa_nodes3 = masked_qa_output2 #qa_nodes2 + masked_qa_output2
        #qa_nodes3 = qa_nodes2 + masked_qa_output2
        qa_output3, embedding_Q3 , qa_A3 = self.qa_encoder3(qa_nodes3, max_question_len, qa_mask, att_bias=qa_A2)
        # print("qa_A3",qa_A3[0,0,0:16,0:16])
        # print("qa_A3",qa_A3[0,1,0:16,0:16])

        self.att_dict["qa_3"] = embedding_Q3

        qa_output = qa_output3*qa_mask
        #qa_output = self.qa_final_ln(qa_output)
        qa_output = qa_output[:,q_rep.shape[2]:,:].view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], 512)
        
        ## reasoning
        reasoning_inp = torch.cat([a_rep, va_output, qa_output], -1)

        reasoning_output = reasoning_inp.view(a_rep.shape[0]*a_rep.shape[1],a_rep.shape[2],1536)
        
        reasoning_nodes = reasoning_output
  
        reasoning_nodes1 = self.reasoning_input_dropout(reasoning_nodes)

        reasoning_output1, _ , reasoning_A1 = self.reasoning_encoder1(reasoning_nodes1, max_answer_len, qa_a_mask)
        # self.att_dict["reason_1"] = reasoning_A1

        masked_reasoning_output1 = reasoning_output1*qa_a_mask
        reasoning_nodes2 = masked_reasoning_output1 #reasoning_nodes1 + masked_reasoning_output1
        #reasoning_nodes2 = reasoning_nodes1 + masked_reasoning_output1

        reasoning_output2, _ , reasoning_A2 = self.reasoning_encoder2(reasoning_nodes2, max_answer_len, qa_a_mask, att_bias=reasoning_A1)
        # self.att_dict["reason_2"] = reasoning_A2

        reasoning_a_rep = reasoning_output2

        #reasoning_a_rep = self.reasoning_final_ln(reasoning_a_rep)
        
        #====================
        masked_pool_rep = reasoning_a_rep*qa_a_mask.float()
        '''
        abs_min_masked_pool_rep = (torch.min(torch.abs(masked_pool_rep), dim=-2)[0]+1e-12).unsqueeze(1)
        
        norm_masked_pool_rep = masked_pool_rep/abs_min_masked_pool_rep
        '''
        score_rep = self.score_fc(masked_pool_rep)
        score_rep = masked_softmax(score_rep, qa_a_mask, dim=-2)
        
        #print("score_rep",score_rep[0,:,:8])
        
        #print("masked_pool_rep",masked_pool_rep.shape)
        
        #print("score_rep:",score_rep.shape)
        
        pool_rep = torch.sum(score_rep*masked_pool_rep, dim=-2).view(a_rep.shape[0], a_rep.shape[1], 1536)
        #========================
        
        # pool_rep = reasoning_a_rep*qa_a_mask.float() #.view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], 1536)
        # pool_true_len = qa_a_mask.sum(dim=1).float()
        # pool_rep = (torch.sum(pool_rep, dim=1)/pool_true_len).view(a_rep.shape[0], a_rep.shape[1], 1536)
        
        #logits = self.final_mlp(pool_rep).squeeze(2)  
        logits = self.final_mlp(pool_rep)#.squeeze(2)
        #print(logits.shape)

        a_dict = self.a(self.adict)
        logitss = self.through_dict(logits,a_dict)
        #scores = torch.sum(options_feat * output_feat, -1)
        #scores = scores.view(batch_size, num_options)
        logits = torch.sum(logits * logitss,-1)
        #print(logits.shape)
        logits = logits.view(logitss.shape[0],logitss.shape[1])
        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)
        option_score = torch.sigmoid(logits)
        
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_lenr = int(box_maskr.sum(1).max().item())
        box_maskr = box_maskr[:, :max_lenr]
        boxesr = boxesr[:, :max_lenr]
        det_featuresr = det_featuresr[:,:max_lenr,:]
        features_2dr = features_2dr[:, :max_lenr]

        obj_repsr = det_featuresr
        obj_repsr = self.obj_downsample(obj_repsr)

        #obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        
        obj_repsr = torch.cat([obj_repsr, features_2dr], dim=-1)
        obj_repsr = self.boxes_fc(obj_repsr)

        # Now get the question representations
        ##q_rep_orir, q_obj_repsr = self.embed_span(question, question_tags, question_mask, obj_repsr)  #
        #a_rep_orir, a_obj_repsr = self.embed_span(answers, answer_tags, answer_mask, obj_repsr)  #

        o_repr = obj_repsr ## [b,max_len,512]

        ## [b,4,len]
        #max_answer_len = int(answer_mask.sum(dim=2).max().item())
        #max_question_len = int(question_mask.sum(dim=2).max().item())

        q_repr = q_rep#q_rep_orir[:,:,:max_question_len,:]
        a_repr = a_rep#a_rep_orir[:,:,:max_answer_len,:]
        #question_mask = question_mask[:,:,:max_question_len]
        #answer_mask = answer_mask[:,:,:max_answer_len]

        ## construct VA graph
        va_v_repr = o_repr.unsqueeze(1).repeat(1,4,1,1)
        va_v_repr = va_v_repr.view(va_v_repr.shape[0] * va_v_repr.shape[1], va_v_repr.shape[2], va_v_repr.shape[3])
        va_a_repr = a_repr.view(a_repr.shape[0] * a_repr.shape[1], a_repr.shape[2], a_repr.shape[3])
        va_nodesr = torch.cat([va_v_repr, va_a_repr],-2) ## [b*4,o_l+a_l, 512]
        
        va_v_maskr = box_maskr.unsqueeze(1).repeat(1,4,1)
        va_v_maskr = va_v_maskr.view(va_v_maskr.shape[0] * va_v_maskr.shape[1], va_v_maskr.shape[2], 1)
        va_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        va_maskr = torch.cat([va_v_maskr, va_a_mask],-2)

        va_nodes1r = self.va_input_dropout(va_nodesr)

        ## va encoder
        va_output1r, embedding_V1r ,va_A1r = self.va_encoder1(va_nodes1r, max_lenr, va_maskr, features_2dr[:,:,:2])

        #self.att_dict["va_1"] = embedding_V1r

        masked_va_output1r = va_output1r*va_maskr
        va_nodes2r = masked_va_output1r #va_nodes1 + masked_va_output1
        #va_nodes2 = va_nodes1 + masked_va_output1
        #va_A1 = 
        va_output2r, embedding_V2r , va_A2r = self.va_encoder2(va_nodes2r, max_lenr, va_maskr, features_2dr[:,:,:2], att_bias=va_A1r)

        #self.att_dict["va_2"] = embedding_V2r

        masked_va_output2r = va_output2r*va_maskr 
        va_nodes3r = masked_va_output2r #va_nodes2 + masked_va_output2
        #va_nodes3 = va_nodes2 + masked_va_output2
        va_output3r, embedding_V3r , va_A3r = self.va_encoder3(va_nodes3r, max_lenr, va_maskr, features_2dr[:,:,:2], att_bias=va_A2r)

        #self.att_dict["va_3"] = embedding_V3r

        va_outputr = va_output3r*va_maskr
        #va_output = self.va_final_ln(va_output)
        va_outputr = va_outputr[:,o_repr.shape[1]:,:].view(a_repr.shape[0], a_repr.shape[1], a_repr.shape[2], 512)
       
        ## construct QA graph
        qa_q_repr = q_repr.view(q_repr.shape[0] * q_repr.shape[1], q_repr.shape[2], q_repr.shape[3])
        qa_a_repr = a_repr.view(a_repr.shape[0] * a_repr.shape[1], a_repr.shape[2], a_repr.shape[3])
        qa_nodesr = torch.cat([qa_q_repr, qa_a_repr],-2) ## [b*4,q_l+a_l, 512]
        
        #qa_q_mask = question_mask.view(question_mask.shape[0] * question_mask.shape[1], question_mask.shape[2], 1)
        #qa_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        #qa_mask = torch.cat([qa_q_mask, qa_a_mask],-2)
        
        qa_nodes1r = self.qa_input_dropout(qa_nodesr)

        ## qa encoder
        qa_output1r, embedding_Q1r , qa_A1r = self.qa_encoder1(qa_nodes1r, max_question_len, qa_mask)
        # print("qa_A1",qa_A1[0,0,0:16,0:16])
        # print("qa_A1",qa_A1[0,1,0:16,0:16])

        #self.att_dict["qa_1"] = embedding_Q1r

        masked_qa_output1r = qa_output1r*qa_mask
        qa_nodes2r = masked_qa_output1r #qa_nodes1 + masked_qa_output1
        #qa_nodes2 = qa_nodes1 + masked_qa_output1
        #va_A1 = 
        qa_output2r, embedding_Q2r , qa_A2r = self.qa_encoder2(qa_nodes2r, max_question_len, qa_mask, att_bias=qa_A1r)
        # print("qa_A2",qa_A2[0,0,0:16,0:16])
        # print("qa_A2",qa_A2[0,1,0:16,0:16])

        #self.att_dict["qa_2"] = embedding_Q2r

        masked_qa_output2r = qa_output2r*qa_mask  
        qa_nodes3r = masked_qa_output2r #qa_nodes2 + masked_qa_output2
        #qa_nodes3 = qa_nodes2 + masked_qa_output2
        qa_output3r, embedding_Q3r , qa_A3r = self.qa_encoder3(qa_nodes3r, max_question_len, qa_mask, att_bias=qa_A2r)
        # print("qa_A3",qa_A3[0,0,0:16,0:16])
        # print("qa_A3",qa_A3[0,1,0:16,0:16])

        #self.att_dict["qa_3"] = embedding_Q3r

        qa_outputr = qa_output3r*qa_mask
        #qa_output = self.qa_final_ln(qa_output)
        qa_outputr = qa_outputr[:,q_repr.shape[2]:,:].view(a_repr.shape[0], a_repr.shape[1], a_repr.shape[2], 512)
        
        ## reasoning
        reasoning_inpr = torch.cat([a_repr, va_outputr, qa_outputr], -1)

        reasoning_outputr = reasoning_inpr.view(a_repr.shape[0]*a_repr.shape[1],a_repr.shape[2],1536)
        
        reasoning_nodesr= reasoning_outputr
  
        reasoning_nodes1r = self.reasoning_input_dropout(reasoning_nodesr)

        reasoning_output1r, _ , reasoning_A1r = self.reasoning_encoder1(reasoning_nodes1r, max_answer_len, qa_a_mask)
        # self.att_dict["reason_1"] = reasoning_A1

        masked_reasoning_output1r = reasoning_output1r*qa_a_mask
        reasoning_nodes2r = masked_reasoning_output1r #reasoning_nodes1 + masked_reasoning_output1
        #reasoning_nodes2 = reasoning_nodes1 + masked_reasoning_output1

        reasoning_output2r, _ , reasoning_A2r = self.reasoning_encoder2(reasoning_nodes2r, max_answer_len, qa_a_mask, att_bias=reasoning_A1r)
        # self.att_dict["reason_2"] = reasoning_A2

        reasoning_a_repr = reasoning_output2r

        #reasoning_a_rep = self.reasoning_final_ln(reasoning_a_rep)
        
        #====================
        masked_pool_repr = reasoning_a_repr*qa_a_mask.float()
        '''
        abs_min_masked_pool_rep = (torch.min(torch.abs(masked_pool_rep), dim=-2)[0]+1e-12).unsqueeze(1)
        
        norm_masked_pool_rep = masked_pool_rep/abs_min_masked_pool_rep
        '''
        score_repr = self.score_fc(masked_pool_repr)
        score_repr = masked_softmax(score_repr, qa_a_mask, dim=-2)
        
        #print("score_rep",score_rep[0,:,:8])
        
        #print("masked_pool_rep",masked_pool_rep.shape)
        
        #print("score_rep:",score_rep.shape)
        
        pool_repr = torch.sum(score_repr*masked_pool_repr, dim=-2).view(a_repr.shape[0], a_repr.shape[1], 1536)
        #========================
        
        '''logits1 = self.final_mlp(pool_repr)#.squeeze(2)
        #print(logits.shape)

        logitss1 = self.through_dict(logits1,a_dict)
        #scores = torch.sum(options_feat * output_feat, -1)
        #scores = scores.view(batch_size, num_options)
        logits1 = torch.sum(logits1 * logitss1,-1)
        #print(logits.shape)
        logits1 = logits1.view(logitss1.shape[0],logitss1.shape[1])'''

        logits1 = self.final_mlp0(pool_repr).squeeze(2)  

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                        "att_dict": self.att_dict
                       }
        if label is not None:
            label_one_hot = label.unsqueeze(-1)
            bce_label = torch.zeros(label.shape[0], 4).cuda()
            bce_label.scatter_(1,label_one_hot,1).cuda()

            option_score = option_score.view(-1)
            bce_label = bce_label.view(-1)

            loss_cls = self.cls_loss(logits, label.long().view(-1))+self.cls_loss(logits1, label.long().view(-1))
            loss_b_cls = self.b_cls_loss(option_score, bce_label)
            self._accuracy(logits, label)
            output_dict["loss"] = loss_cls+loss_b_cls
            output_dict["cls_loss"] = loss_cls
            output_dict["b_cls_loss"] = loss_b_cls
            output_dict["label"] = label.long().view(-1)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
