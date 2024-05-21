import os
import cv2
import torch
import torch.nn.functional as F

from torchvision import transforms
from comer.datamodule.vocab import vocab
from comer.tra2img import *
from comer.model.encoder import Encoder
from comer.model.decoder import Decoder
import onnxruntime
from collections import OrderedDict
import numpy

def model_load(encoder_weight_path, decoder_weight_path):
    encoder = Encoder(256, 24, 16)
    encoder.load_state_dict(torch.load(encoder_weight_path))
    encoder.eval()
    
    decoder = Decoder(256, 8, 3, 1024, 0.3, 32, True, False)
    decoder.load_state_dict(torch.load(decoder_weight_path))
    decoder.eval()
    
    return encoder, decoder

def trans_image_tensor(data):
    trans_list = [transforms.ToTensor()]
    transform = transforms.Compose(trans_list)
    return transform(data)
    
def formula_recog(trace, encoder, decoder):
    img = trace2image(trace)
    cv2.imwrite("./tmp.jpg", img)
    input = trans_image_tensor(img)
    input = input.unsqueeze(0)
    mask = torch.zeros(1, input[0].size(1), input[0].size(2), dtype=torch.bool)


    
    device = 'cpu' # cuda:0, cpu
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    feature, mask = encoder(input.to(device), mask.to(device))
    # input_xx = torch.randn(1,1,200,800)
    # input_mask = torch.zeros(1,200,800,dtype=torch.bool)
    # torch.onnx.export(encoder, (input_xx.to(device), input_mask.to(device)), "encoder_0508.onnx", input_names=["input", "inp_mask"], output_names=["feature", "oup_mask"])
    src = [feature]
    src_mask = [mask]
    batch_size = src[0].shape[0] * 2  # mul 2 for bi-direction decode
    half_bb_size = batch_size // 2
    for i in range(len(src)):
        src[i] = torch.cat((src[i], src[i]), dim=0)
        src_mask[i] = torch.cat((src_mask[i], src_mask[i]), dim=0)

    l2r = torch.full(
        (batch_size // 2, 1),
        fill_value=vocab.SOS_IDX,
        dtype=torch.int32,
        device='cpu',
    )
        
    r2l = torch.full(
        (batch_size // 2, 1),
        fill_value=vocab.EOS_IDX,
        dtype=torch.int32,
        device='cpu',
    )
    input_ids = torch.cat((l2r, r2l), dim=0)
    
    # hyps: result of bi-direction decode
    # scores: conditional probability of bi-direction decode
    _, cur_len = input_ids.shape # 2, 1
    l2r_score = torch.ones(1, device='cpu',)
    r2l_score = torch.ones(1, device='cpu',)
    
    # print(src[0].shape)

    # srcList=[]
    
    # with open('1.log', 'r') as file:
    #     for line in file:
    #         srcList.append((float)(line))
    
    # pysrcList = []

    # for i in range(len(src[0])):
    #     for j in range(len(src[0][i])):
    #         for k in range(len(src[0][i][j])):
    #             for l in range(len(src[0][i][j][k])):
    #                 pysrcList.append((float)(src[0][i][j][k][l].item()))
    
    # print(len(srcList))
    # print(len(pysrcList))
    
    # np.testing.assert_allclose(srcList, pysrcList, rtol=5e-04, atol=5e-04)
    max_len = 50
    while cur_len < max_len:

        # session = onnxruntime.InferenceSession("./decoder_0520_ds.onnx", providers=['CPUExecutionProvider'])
        # input_name= [session.get_inputs()[0].name]
        # input_name.append(session.get_inputs()[1].name)
        # input_name.append(session.get_inputs()[2].name)
        # outputs = [x.name for x in session.get_outputs()]
        # print("onnx input_name:", input_name)
        # print("onnx outputs:", outputs)

        # # exit(1)
        # ort_outs = session.run(None, {"input1": src[0].detach().numpy(), "input2": src_mask[0].detach().numpy(), "input3": input_ids.detach().numpy()})
        # # ort_outs = session.run(outputs, {input_name: input_data})
        # ort_outs = OrderedDict(zip(outputs, ort_outs))
    
        # # For debug
        # for key in ort_outs:
        #     val = ort_outs[key]
        #     print(val)
        
        print("------------------------------------")
        print(src[0].shape)
        print(src_mask[0].shape)
        print("------------------------------------")
        
        next_token_logits = (decoder(src[0].to(device), src_mask[0].to(device), input_ids.to(device))[:, -1, :])
        # next_token_logits = (decoder(torch.from_numpy(src_inp).to(device), torch.from_numpy(mask_inp).to(device), torch.from_numpy(src_inp).to(device))[:, -1, :])

        # decoder.eval()
        # trace
        # decoder = torch.jit.trace(decoder, (src[0].to(device), src_mask[0].to(device), input_ids.to(device)))
        # decoder.save('decoder.pt')
        # script
        # model_script = torch.jit.script(model)
        # model_script.save('model_script.pt')

        torch.onnx.export(decoder, (src[0].to(device), src_mask[0].to(device), input_ids.to(device)),
                     "decoder_0521_ds.onnx", input_names=["input1","input2","input3"], output_names=["output"],
                    dynamic_axes={"input1":{2:"input_width"},"input2":{2:"input_width"}, "input3":{1:"length"}},
                    verbose=False, opset_version=13)
        break
        next_token_scores = F.softmax(next_token_logits, dim=-1)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 1, dim=1
        )
        # next_token_scores, next_tokens = next_token_scores.cuda(), next_tokens.cuda()
        next_token_scores, next_tokens = next_token_scores, next_tokens
        
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        # print(input_ids)
        if (2 in input_ids[0]) and (1 in input_ids[1]): # 双向解码均结束时退出循环
            break
        # cal conditional probability of l2r decode
        if 2 not in input_ids[0]:
            l2r_score *= next_token_scores[0]
        # cal conditional probability of r2l decode
        if 1 not in input_ids[1]:
            r2l_score *= next_token_scores[1]
        
        cur_len += 1

    hyps, scores = [input_ids[0][1:-1], input_ids[1][1:-1]], torch.cat((l2r_score, r2l_score), dim = 0)
    


    for i in range(half_bb_size, batch_size):
        hyps[i] = torch.flip(hyps[i], dims=[0])
    
    if scores[0] >= scores[1]:
        out = hyps[0]
    else:
        out = hyps[1]
    
    result = vocab.indices2label(out.tolist()) # id -> char
    result = result.replace('<eos>', '')
    result = result.replace('<sos>', '')
    if ('lg' in result) and ('\lg' not in result):
        result = result.replace('lg', '\lg')
    if ('ln' in result) and ('\ln' not in result):
        result = result.replace('ln', '\ln')
    result = result.replace(' ', '')
    
    return result
    

if __name__ == '__main__':
    encoder_weight_path = './encoder.pth'
    decoder_weight_path = './decoder.pth'
    encoder, decoder = model_load(encoder_weight_path, decoder_weight_path)

    json_path = 'data/handwriting_res_2022_06_20_09_24_11.json'
    with open(json_path, 'r') as f:
        libread = json.load(f)
    points = libread['points']
    gt_label = libread['predict']
        
    trace = trace_transform_json(points)
    pre_result = formula_recog(trace, encoder, decoder)
    print('gt: ', gt_label)
    print('pre: ', pre_result)
