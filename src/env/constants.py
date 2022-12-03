state_names = [
    # talk 8 | total 8
    "s_talk_bry", "s_talk_eli", "s_talk_ext", "s_talk_for", "s_talk_kim",
    "s_talk_que", "s_talk_rob", "s_talk_ter",
    # poster 22 | total 30
    "s_post_ali", "s_post_ant", "s_post_bac", "s_post_bac_rep", "s_post_bac_str",
    "s_post_bac_dis", "s_post_bot", "s_post_ebo", "s_post_ele_mic", "s_post_how_pat",
    "s_post_inf", "s_post_opt_mic", "s_post_pat_mut", "s_post_pre_tre", "s_post_sal",
    "s_post_sci_met", "s_post_siz_com", "s_post_sma", "s_post_vir_dis", "s_post_vir_rep",
    "s_post_vir_str", "s_post_vir",
    # book 9 | total 39
    "s_book_ant", "s_book_bac", "s_book_bot", "s_book_ebo", "s_book_fun",
    "s_book_inf", "s_book_sal", "s_book_sma", "s_book_vir",
    # objects 17 | total 56
    "s_obj_app", "s_obj_ban", "s_obj_bre", "s_obj_cac", "s_obj_che",
    "s_obj_coc", "s_obj_egg", "s_obj_mil", "s_obj_oj", "s_obj_ora",
    "s_obj_pie", "s_obj_san", "s_obj_wat", "s_obj_ket", "s_obj_jar10",
    "s_obj_jar11", "s_obj_jar4",
    # objects tested 17 | total 73
    "s_objtest_app", "s_objtest_ban", "s_objtest_bre", "s_objtest_cac", "s_objtest_che",
    "s_objtest_coc", "s_objtest_egg", "s_objtest_mil", "s_objtest_oj", "s_objtest_ora",
    "s_objtest_pie", "s_objtest_san", "s_objtest_wat", "s_objtest_ket", "s_objtest_jar10",
    "s_objtest_jar11", "s_objtest_jar4",
    # individuals | total 83
    "s_testpos", "s_testleft", "s_label", "s_label_lesson", "s_label_slide",
    "s_worksheet", "s_workshsubmit", "s_notetake", "s_noteview", "s_computer",
    # aes 14 | total 97
    "s_aes_bry_1", "s_aes_bry_2", "s_aes_ter_1", "s_aes_ter_2", "s_aes_ter_3",
    "s_aes_kno_1", "s_aes_kno_2", "s_aes_wor_1", "s_aes_wor_2", "s_aes_wor_3",
    "s_aes_pas_1", "s_aes_pas_2", "s_aes_que_1", "s_aes_que_2",
    # player static attributes | total 100
    "s_static_gender", "s_static_pretest", "s_static_gameskill",
    # target goal | total 102
    "s_target_disease", "s_target_item",
    # solved and steps | total 104
    "s_solved", "s_end", "s_step"
]

action_names = [
    # talk 8 | total 8
    "a_talk_bry", "a_talk_eli", "a_talk_ext", "a_talk_for", "a_talk_kim",
    "a_talk_que", "a_talk_rob", "a_talk_ter",
    # individual 12 | total 20
    "a_obj", "a_objtest", "a_book", "a_post", "a_notetake",
    "a_noteview", "a_computer", "a_worksheet", "a_label", "a_testleft",
    "a_workshsubmit", "a_end"
]

state_map = {name:i for (i,name) in enumerate(state_names)}
action_map = {name:i for (i,name) in enumerate(action_names)}
state_map_rev = {i:name for (i,name) in enumerate(state_names)}
action_map_rev = {i:name for (i,name) in enumerate(action_names)}


# obj pickup probability
s_obj_p = {
    's_obj_app': 0.012763204328902482, 's_obj_ban': 0.09987060345841665, 's_obj_bre': 0.114163039642395,
    's_obj_cac': 0.013292553817197976, 's_obj_che': 0.02623220797553229, 's_obj_coc': 0.06393365486413363,
    's_obj_egg': 0.09781202211504529, 's_obj_jar10': 0.025526408657804964, 's_obj_jar11': 0.023173744265380544,
    's_obj_jar4': 0.026585107634395953, 's_obj_ket': 0.009881190448182567, 's_obj_mil': 0.17662627926126337,
    's_obj_oj': 0.06034584166568639, 's_obj_ora': 0.027996706269850607, 's_obj_pie': 0.042583225502882016,
    's_obj_san': 0.10722267968474297, 's_obj_wat': 0.07199153040818727
}

max_ep_len = 230  # 90th percentile is 227