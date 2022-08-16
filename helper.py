from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter


import emojis

extract = URLExtract()

def fetch_stats(selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df=df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]



    num_messages=df.shape[0]

    words = []
    for messages in df['messages']:
        words.extend(messages.split())

    num_media_messsages = df[df['messages'] == '<Media omitted>'].shape[0]

    links = []
    for message in df['messages']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messsages,len(links)



def monthly_timeline (selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]


    timeline = df.groupby(['year', 'month_num', 'month']).count()['messages'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]


    daily_timeline = df.groupby('only_date').count()['messages'].reset_index()

    return daily_timeline


def week_activity_map(selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]


    return df['month'].value_counts()

def activity_heatmap(selected_user,df,selected_sentiment):

    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]


    user_heatmap = df.pivot_table(index='day_name', columns='period', values='messages', aggfunc='count').fillna(0)

    return user_heatmap


def most_busy_users(df,selected_sentiment):
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]

    x = df['users'].value_counts().head()
    new_df = round((df['users'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'users': 'percent'})
    return x,new_df

def create_wordcloud(selected_user,df,selected_sentiment):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user!='Overall':
        df=df[df['users']==selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]

    df=df[df['messages']!='Media omitted']
    df=df[df['messages']!='<Media omitted>']




    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words and word.isalpha():
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df['messages'] = df['messages'].apply(remove_stop_words)
    df_wc = wc.generate(df['messages'].str.cat(sep=" "))
    return df_wc

    return df_wc

def most_common_words(selected_user,df,selected_sentiment):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    df=df[df['messages']!='Media omitted']
    df=df[df['messages']!='<Media omitted>']
    df=df[df['messages']!='<Media Omitted>']



    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]



    words=[]

    for message in df['messages']:
        for word in message.lower().split():
            if word not in stop_words and word.isalpha():
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df,selected_sentiment):
    if selected_user != 'Overall':
        df = df[df['users'] == selected_user]
    if selected_sentiment != 'Overall':
        df=df[df['predict'] == selected_sentiment]


    emojiss = []
    for message in df['messages']:
        emojiss.extend(emojis.get(message))

    emoji_df = pd.DataFrame(Counter(emojiss).most_common(5))

    return emoji_df


