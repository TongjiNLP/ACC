from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn

class T5GenerationModel(nn.Module):
    def __init__(self, model_path):
        super(T5GenerationModel, self).__init__()
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        dim = self.t5_model.config.d_model
        n_gpu = torch.cuda.device_count()
        if n_gpu > 0:
            layer_num = self.t5_model.config.num_layers
            layer_per_gpu = layer_num // n_gpu
            layer_per_gpu_remainder = layer_num % n_gpu
            device_map = {}
            cur_layer = 0
            for n in range(n_gpu):
                device_map[n] = []
                if n < layer_per_gpu_remainder:
                    layer_assigned = layer_per_gpu + 1
                else:
                    layer_assigned = layer_per_gpu

                for i in range(layer_assigned):
                    device_map[n].append(cur_layer)
                    cur_layer += 1
            self.t5_model.parallelize(device_map)

    def forward(self, input_ids, input_masks, labels=None):
        if labels is not None:
            t5_output = self.t5_model(input_ids=input_ids,
                                      attention_mask=input_masks,
                                      labels=labels,
                                      return_dict=True)
            loss = t5_output.loss
            return loss
        else:
            # enc_time_beg = time.time()

            # enc_time_end = time.time()
            # dec_time_beg = time.time()
            t5_output = self.t5_model.generate(
                input_ids=input_ids,
                # encoder_outputs=ModelOutput(last_hidden_state=encoder_q),
                max_length = 75,
                attention_mask=input_masks,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=False
            )
            output_sequences = t5_output.sequences
            return output_sequences
            # score_list = t5_output.score_list
            # predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            # predicts = predicts[0].split(split_symbol)
            # dec_time_end = time.time()
            # return predicts, enc_time_end - enc_time_beg, dec_time_end - dec_time_beg