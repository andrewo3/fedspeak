import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./llama/llama-2-7b-chat-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device,torch.version.cuda)


access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_dir,token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_dir,token=access_token,device_map="auto",torch_dtype = torch.bfloat16)

prompt = """Consider the following passage (paragraph numbers denoted by [paragraph #]), adapted from John F. Kennedy’s Inaugural
address delivered Friday,
January 20, 1961, and answer the questions:
"[1] So let us begin a new remembering on both sides that civility is not a sign of weakness, and sincerity is always
subject to proof. Let us never negotiate out of fear. But let us never fear to negotiate. And if a beachhead of
cooperation may push back the jungle of suspicion, let both sides join in creating a new endeavor, not a new balance of
power, but a new world of law, where the strong are just and the weak secure and the peace preserved. [2] All this will
not be finished in the first one hundred days. Nor will it be finished in the first one thousand days, nor in the life
of this Administration, nor even perhaps in our lifetime on this planet. But let us begin. [3] In your hands, my fellow
citizens, more than mine, will rest the final success or failure of our course. Since this country was founded, each
generation of Americans has been summoned to give testimony to its national loyalty. The graves of young Americans who
answered the call to service surround the globe. [4] Now the trumpet summons us again; not as a call to bear arms,
though arms we need; not as a call to battle, though embattled we are; but a call to bear the burden of a long twilight
struggle, year in and year out, “rejoicing in hope, patient in tribulation”—a struggle against the common enemies of
man: tyranny, poverty, disease and war itself. Can we forge against these enemies a grand and global alliance, North and
 South, East and West, that can assure a more fruitful life for all mankind? Will you join in that historic effort? [5]
 In the long history of the world, only a few generations have been granted the role of defending freedom in its hour of
  maximum danger. I do not shrink from this responsibility; I welcome it. I do not believe that any of us would exchange
   places with any other people or any other generation. The energy, the faith, the devotion which we bring to this
   endeavor will light our country and all who serve it—and the glow from that fire can truly light the world. [6] And
   so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country. My fellow
   citizens of the world: ask not what America will do for you, but what together we can do for the freedom of man.
   [7] Finally, whether you are citizens of America or citizens of the world, ask of us here the same high standards of
   strength and sacrifice which we ask of you. With a good conscience our only sure reward, with history the final judge of our deeds, let us go forth to lead the land we love, asking His blessing and His help, but knowing that here on Earth God’s work must truly be our own."?"""
prompt+="\n\n"
questions = ["Upon reading this passage, what is the central idea that Kennedy expresses in this speech?",
             "What does the second paragraph serve to do?",
             "According to Kennedy, what is somewhat unique to this generation of Americans? Provide evidence."]


for q in questions:
    messages = [{
        "role":"user",
        "content":prompt+q
    }]

    model_inputs = tokenizer.apply_chat_template(messages,return_tensors = "pt").to(device)

    out = model.generate(model_inputs,max_new_tokens = 1000, do_sample=True,pad_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.batch_decode(out)

    print(decoded[0][decoded[0].index("[/INST]"):])