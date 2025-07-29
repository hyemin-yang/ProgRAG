from prompts import *

def first_prompt_matching(cnt, original_q, topic_ent, writer):
    if cnt == 1:
        input_text = PROMPT_FIRST_SPLIT.format(Q=original_q, topic=topic_ent)
    else:
        if type(topic_ent) == list:
            input_text = PROMPT_LISTNEXT_SPLIT.format(Q=original_q, Sub_Q= writer.sub_questions, topic=topic_ent)
        else:
            if len(topic_ent) > 1 and topic_ent[1] != '.':
                input_text = PROMPT_NEXT_SPLIT.format(Q=original_q, Sub_Q= writer.sub_questions, topic=topic_ent)
            else:
                input_text = PROMPT_IDNEXT_SPLIT.format(Q=original_q, Sub_Q= writer.sub_questions, topic=topic_ent)
    
    return input_text

def rel_prompt_mathcing(args, topic_ent, sub_Q, writer, cand_rel, out_form):
    if args.dataset == 'cwq':
        input_text = CWQ_TOPK_PROMPT_REL_FIND.format(Q=sub_Q, topic=topic_ent, candidate_rels=cand_rel)
    else:
        input_text = REVISED_TOPK_PROMPT_REL_FIND.format(Q=sub_Q, topic=topic_ent, candidate_rels=cand_rel, typeof=out_form)

    return input_text