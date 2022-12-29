state_names = [
    # talk 8 | total 8
    "s_talk_bryce", "s_talk_elise", "s_talk_extra", "s_talk_ford", "s_talk_kim",
    "s_talk_quentin", "s_talk_robert", "s_talk_teresa",
    # poster 22 | total 30
    "s_post_alive", "s_post_anthrax", "s_post_bacteria", "s_post_bacteria_reproduction", "s_post_bacteria_structure",
    "s_post_bacterial_diseases", "s_post_botulism", "s_post_ebola", "s_post_electron_microscope", "s_post_how_pathogens",
    "s_post_influenza", "s_post_optical_microscope", "s_post_pathogens_mutagens", "s_post_prevention_treatment", "s_post_salmonellosis",
    "s_post_scientific_method", "s_post_size_comparison", "s_post_smallpox", "s_post_viral_diseases", "s_post_virus_reproduction",
    "s_post_virus_structure", "s_post_viruses",
    # book 9 | total 39
    "s_book_anthrax", "s_book_bacteria", "s_book_botulism", "s_book_ebola", "s_book_fungi",
    "s_book_influenza", "s_book_salmonellosis", "s_book_smallpox", "s_book_viruses",
    # objects 17 | total 56
    "s_obj_apple", "s_obj_bananas", "s_obj_bread", "s_obj_cactus", "s_obj_cheese",
    "s_obj_coconut", "s_obj_egg", "s_obj_milk", "s_obj_oj", "s_obj_orange",
    "s_obj_pie", "s_obj_sandwich", "s_obj_water", "s_obj_kettle", "s_obj_jar10",
    "s_obj_jar11", "s_obj_jar4",
    # objects tested 17 | total 73
    "s_objtest_apple", "s_objtest_bananas", "s_objtest_bread", "s_objtest_cactus", "s_objtest_cheese",
    "s_objtest_coconut", "s_objtest_egg", "s_objtest_milk", "s_objtest_oj", "s_objtest_orange",
    "s_objtest_pie", "s_objtest_sandwich", "s_objtest_water", "s_objtest_kettle", "s_objtest_jar10",
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
    # solved and end | total 104
    "s_solved", "s_end"
]

action_names = [
    # talk 8 | total 8
    "a_talk_bryce", "a_talk_elise", "a_talk_extra", "a_talk_ford", "a_talk_kim",
    "a_talk_quentin", "a_talk_robert", "a_talk_teresa",
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
    's_obj_apple': 0.012763204328902482, 's_obj_bananas': 0.09987060345841665, 's_obj_bread': 0.114163039642395,
    's_obj_cactus': 0.013292553817197976, 's_obj_cheese': 0.02623220797553229, 's_obj_coconut': 0.06393365486413363,
    's_obj_egg': 0.09781202211504529, 's_obj_jar10': 0.025526408657804964, 's_obj_jar11': 0.023173744265380544,
    's_obj_jar4': 0.026585107634395953, 's_obj_kettle': 0.009881190448182567, 's_obj_milk': 0.17662627926126337,
    's_obj_oj': 0.06034584166568639, 's_obj_orange': 0.027996706269850607, 's_obj_pie': 0.042583225502882016,
    's_obj_sandwich': 0.10722267968474297, 's_obj_water': 0.07199153040818727
}

max_ep_len = 230  # 95th percentile is 230
min_ep_len = 50  # 05th percentile is 97
# action_probs = print(list(df_org.groupby('action').count()['step'] / len(df_org)))
action_probs = [0.02372234935163997, 0.03642802658820966, 0.03030402092186989, 0.020605862482292688, 0.03309360357415277, 0.02596709164214885, 0.021412226217718208, 0.02384221423123025, 0.18568159529257927, 0.055519232864770625, 0.031012313392176093, 0.23595946387708402, 0.03342050779121717, 0.03801896044458974, 0.02637027350986161, 0.05332897461043914, 0.04886128364389234, 0.03573063092513894, 0.021978860193963168, 0.018742508445025608]
# use util function utils.get_action_probs()
obj_post_book_probs = {'a_objtest': {'s_objtest_app': 0.013038324772817068, 's_objtest_ban': 0.10885025681548795, 's_objtest_bre': 0.151521137890162, 's_objtest_cac': 0.0039510075069142635, 's_objtest_che': 0.022323192414065586, 's_objtest_coc': 0.0705254839984196, 's_objtest_egg': 0.1291979454760964, 's_objtest_jar10': 0.007111813512445673, 's_objtest_jar11': 0.007902015013828527, 's_obj_jar4': 0.005333860134334255, 's_objtest_ket': 0.002765705254839984, 's_objtest_mil': 0.1710786250493876, 's_objtest_oj': 0.05906756222836823, 's_objtest_ora': 0.040300276570525484, 's_obj_pie': 0.028249703674436983, 's_objtest_san': 0.09541683129197945, 's_objtest_wat': 0.08336625839589096},
                       'a_obj': {'s_obj_apple': 0.012763204328902482, 's_obj_bananas': 0.09987060345841665, 's_obj_bread': 0.114163039642395, 's_obj_cactus': 0.013292553817197976, 's_obj_cheese': 0.02623220797553229, 's_obj_coconut': 0.06393365486413363, 's_obj_egg': 0.09781202211504529, 's_obj_jar10': 0.025526408657804964, 's_obj_jar11': 0.023173744265380544, 's_obj_jar4': 0.026585107634395953, 's_obj_kettle': 0.009881190448182567, 's_obj_milk': 0.17662627926126337, 's_obj_oj': 0.06034584166568639, 's_obj_orange': 0.027996706269850607, 's_obj_pie': 0.042583225502882016, 's_obj_sandwich': 0.10722267968474297, 's_obj_water': 0.07199153040818727},
                       'a_book': {'s_book_anthrax': 0.08450704225352113, 's_book_bacteria': 0.16255868544600938, 's_book_botulism': 0.08626760563380281, 's_book_ebola': 0.0886150234741784, 's_book_fungi': 0.1073943661971831, 's_book_influenza': 0.12852112676056338, 's_book_salmonellosis': 0.12793427230046947, 's_book_smallpox': 0.0897887323943662, 's_book_viruses': 0.12441314553990611},
                       'a_post': {'s_post_alive': 0.040377889812417675, 's_post_anthrax': 0.03015851387564155, 's_post_bacteria': 0.031157741745015215, 's_post_bacterial_diseases': 0.03311077803515465, 's_post_bacteria_reproduction': 0.0327474224462915, 's_post_bacteria_structure': 0.09928691465685606, 's_post_botulism': 0.039969114774946635, 's_post_ebola': 0.016441840396057592, 's_post_electron_microscope': 0.09469955034745878, 's_post_how_pathogens': 0.012263251124131354, 's_post_influenza': 0.045464868056501795, 's_post_optical_microscope': 0.021755915883181178, 's_post_pathogens_mutagens': 0.0685379479493119, 's_post_prevention_treatment': 0.09638006994595086, 's_post_salmonellosis': 0.03229322796021256, 's_post_scientific_method': 0.14793114411591043, 's_post_size_comparison': 0.034337103147567785, 's_post_smallpox': 0.03306535858654676, 's_post_viruses': 0.015260934732252351, 's_post_viral_diseases': 0.011036926011718217, 's_post_virus_reproduction': 0.015896807012762866, 's_post_virus_structure': 0.047826679384112274}}


# d = df.loc[(df['action']==envconst.action_map['a_label']) & (df['reward']!=0)].copy()
# d['temp1'] = d.apply(lambda x: x['info'].get('extra', (-1, -1))[0], axis=1)
# d['temp2'] = d.apply(lambda x: x['info'].get('extra', (-1, -1))[1], axis=1)
label_accept = {'s_label_slide': [.27, .73], 's_label_lesson': [.12, .88]}
