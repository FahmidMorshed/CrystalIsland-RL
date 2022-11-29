state_names = [
    # location 5 | total 5
    "s_loc_bry", "s_loc_din", "s_loc_inf", "s_loc_lab", "s_loc_men",
    # talk 8 | total 13
    "s_talk_bry", "s_talk_eli", "s_talk_ext", "s_talk_for", "s_talk_kim",
    "s_talk_que", "s_talk_rob", "s_talk_ter",
    # poster 22 | total 35
    "s_post_ali", "s_post_ant", "s_post_bac", "s_post_bac_rep", "s_post_bac_str",
    "s_post_bac_dis", "s_post_bot", "s_post_ebo", "s_post_ele_mic", "s_post_how_pat",
    "s_post_inf", "s_post_opt_mic", "s_post_pat_mut", "s_post_pre_tre", "s_post_sal",
    "s_post_sci_met", "s_post_siz_com", "s_post_sma", "s_post_vir_dis", "s_post_vir_rep",
    "s_post_vir_str", "s_post_vir",
    # book 9 | total 44
    "s_book_ant", "s_book_bac", "s_book_bot", "s_book_ebo", "s_book_fun",
    "s_book_inf", "s_book_sal", "s_book_sma", "s_book_vir",
    # objects 17 | total 61
    "s_obj_app", "s_obj_ban", "s_obj_bre", "s_obj_cac", "s_obj_che",
    "s_obj_coc", "s_obj_egg", "s_obj_mil", "s_obj_oj", "s_obj_ora",
    "s_obj_pie", "s_obj_san", "s_obj_wat", "s_obj_ket", "s_obj_jar10",
    "s_obj_jar11", "s_obj_jar4",
    # objects tested 17 | total 78
    "s_objtest_app", "s_objtest_ban", "s_objtest_bre", "s_objtest_cac", "s_objtest_che",
    "s_objtest_coc", "s_objtest_egg", "s_objtest_mil", "s_objtest_oj", "s_objtest_ora",
    "s_objtest_pie", "s_objtest_san", "s_objtest_wat", "s_objtest_ket", "s_objtest_jar10",
    "s_objtest_jar11", "s_objtest_jar4",
    # individuals | total 89
    "s_testpos", "s_testleft", "s_label", "s_label_lesson", "s_label_slide",
    "s_worksheet", "s_workshsubmit", "s_notetake", "s_noteview", "s_computer",
    "s_quiz",
    # aes 14 | total 103
    "s_aes_bry_1", "s_aes_bry_2", "s_aes_ter_1", "s_aes_ter_2", "s_aes_ter_3",
    "s_aes_kno_1", "s_aes_kno_2", "s_aes_wor_1", "s_aes_wor_2", "s_aes_wor_3",
    "s_aes_pas_1", "s_aes_pas_2", "s_aes_que_1", "s_aes_que_2",
    # player static attributes | total 106
    "s_static_gender", "s_static_pretest", "s_static_gameskill",
    # target goal | total 108
    "s_target_disease", "s_target_item"
]

action_names = [
    # location 5 | total 5
    "a_loc_bry", "a_loc_din", "a_loc_inf", "a_loc_lab", "a_loc_men",
    # talk 8 | total 13
    "a_talk_bry", "a_talk_eli", "a_talk_ext", "a_talk_for", "a_talk_kim",
    "a_talk_que", "a_talk_rob", "a_talk_ter",
    # individual 11 | total 24
    "a_obj", "a_objtest", "a_book", "a_post", "a_notetake",
    "a_noteview", "a_computer", "a_worksheet", "a_workshsubmit", "a_label",
    "a_testleft",
]

state_map = {name:i for (i,name) in enumerate(state_names)}
action_map = {name:i for (i,name) in enumerate(action_names)}
state_map_rev = {i:name for (i,name) in enumerate(state_names)}
action_map_rev = {i:name for (i,name) in enumerate(action_names)}

post_loc_map = {
    "s_post_ali": "s_loc_lab", "s_post_ant": "s_loc_inf", "s_post_bac": "s_loc_men", "s_post_bac_rep": "s_loc_men", "s_post_bac_str": "s_loc_lab",
    "s_post_bac_dis": "s_loc_lab", "s_post_bot": "s_loc_inf", "s_post_ebo": "s_loc_inf", "s_post_ele_mic": "s_loc_lab", "s_post_how_pat": "s_loc_men",
    "s_post_inf": "s_loc_inf", "s_post_opt_mic": "s_loc_lab", "s_post_pat_mut": "s_loc_lab", "s_post_pre_tre": "s_loc_inf", "s_post_sal": "s_loc_inf",
    "s_post_sci_met": "s_loc_inf", "s_post_siz_com": "s_loc_lab", "s_post_sma": "s_loc_inf", "s_post_vir_dis": "s_loc_men", "s_post_vir_rep": "s_loc_men",
    "s_post_vir_str": "s_loc_lab", "s_post_vir": "s_loc_men",
}
# all books are at s_loc_lab
talk_loc_map = {
    "s_talk_bry": "s_loc_bry",
    "s_talk_eli": "s_loc_lab",
    "s_talk_ext": "s_loc_inf",
    "s_talk_for": "s_loc_men",
    "s_talk_kim": "s_loc_inf",
    "s_talk_que": "s_loc_din",
    "s_talk_rob": "s_loc_men",
    "s_talk_ter": "s_loc_inf",
}

obj_loc_map = {
    "s_obj_app": "s_loc_din", "s_obj_ban": "s_loc_din", "s_obj_bre": "s_loc_lab", "s_obj_cac": "s_loc_bry", "s_obj_che": "s_loc_din",
    "s_obj_coc": "s_loc_lab", "s_obj_egg": "s_loc_din", "s_obj_mil": "s_loc_din", "s_obj_oj": "s_loc_din", "s_obj_ora": "s_loc_din",
    "s_obj_pie": "s_loc_din", "s_obj_san": "s_loc_din", "s_obj_wat": "s_loc_din", "s_obj_ket": "s_loc_din", "s_obj_jar10": "s_loc_inf",
    "s_obj_jar11": "s_loc_inf", "s_obj_jar4": "s_loc_inf",
}

constants = {'state_map': state_map, 'action_map': action_map, 'post_loc_map': post_loc_map, 'talk_loc_map': talk_loc_map, 'obj_loc_map': obj_loc_map, 'state_map_rev': state_map_rev, 'action_map_rev': action_map_rev}