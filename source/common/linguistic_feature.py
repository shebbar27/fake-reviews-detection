def count_pronouns(words, pronouns):
    return len([x for x in words if x in pronouns])

def linguistic_features(df):
    df_new = df.copy()
    df_new['num_word'] = df_new['Word_List_all'].apply(len)
    df_new['num_coreword'] = df_new['Word_List'].apply(len)
    df_new['num_stopword'] = df_new['num_word'] - df_new['num_coreword']
    df_new['num_char'] = df_new['Ori_Review'].apply(len)
    df_new['char_per_word'] = df_new['num_char'] / df_new['num_word']
    first_sing = ['i', 'me', 'my', 'mine', 'myself']
    first_pru = ['we', 'our', 'us', 'ours', 'ourselves']
    third = ['he', 'she', 'they', 'his', 'her', 'their', 'hers', 'theirs', 'himself', 'herself', 'themselves']
    df_new['num_first_sing'] = df_new['Word_List_all'].apply(lambda x: count_pronouns(x, first_sing))
    df_new['num_first_pru'] = df_new['Word_List_all'].apply(lambda x: count_pronouns(x, first_pru))
    df_new['num_third'] = df_new['Word_List_all'].apply(lambda x: count_pronouns(x, third))
    return df_new
