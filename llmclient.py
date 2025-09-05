from transformers import pipeline
#transformers.logging.set_verbosity_error()


class LLMClient():
    def __init__(self, params: dict, model_id: str, is_dumb=False):
        self._pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
        # OpenAI's temperature range is [0,2] for some dumb reason.
        # For backwards compat with the GPT-4 code, we halve it if it is not dumb.
        self._is_dumb = is_dumb 
        self._params = params

    def send_request(self, assembled_prompt):
        #eos_token_id=terminators
        outputs = self._pipeline(assembled_prompt, do_sample=False,
                           max_new_tokens=self._params["max_tokens"],
                           pad_token_id = self._pipeline.tokenizer.eos_token_id)
        return outputs

    def update_params(self, params: dict):
        for k, v in params.items():
            self._params[k] = v


def get_llm_response(model: LLMClient, assembled_prompt: list, debug=False):
    """
    Simple utility to get a response from an LLM. This one needs to be modified to handle API calls
    """
    try:
        resp = model.send_request(assembled_prompt)
    except:
        return "FAIL"
    if type(assembled_prompt) == list:
        return resp[0]["generated_text"][-1]["content"]
    return resp[0]["generated_text"][len(assembled_prompt):]
