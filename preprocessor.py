import pandas as pd
import re
from datetime import datetime as dt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

cv= pickle.load(open('cv1.pkl','rb'))
mnb= pickle.load(open('mnb1.pkl','rb'))



def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{1,4},\s\d{1,2}:\d{1,2}\s[am|pm]+'
    if type(re.split(pattern, data)) == list:
        msg = re.split(pattern, data)[1:]
        time = re.findall(pattern, data)
        df = pd.DataFrame({'time': time, "message": msg})
        users = []
        messages = []
        for message in df['message']:
            entry = re.split('([\w\W]+?):\s', message)
            if entry[1:]:
                users.append(entry[1])
                messages.append(entry[2])
            else:
                users.append('group notification')
                messages.append(entry[0])

        df['users'] = users
        df['messages'] = messages
        df['messages'] = df['messages'].apply(lambda x: x.strip('\n'))
        df['users'] = df['users'].apply(lambda x: x.split('-')[-1])
        df['am_pm'] = df['time'].apply(lambda x: x.split(' ')[-1])
        df['time'] = df['time'].apply(lambda x: x.split(' ')[0]) + df['time'].apply(lambda x: x.split(' ')[1])
        df['time'] = pd.to_datetime(df['time'], format='%d/%m/%y,%H:%M')

        df = df[df['users'] != 'group notification']
        df['month'] = df['time'].dt.month_name()
        df['year'] = df['time'].dt.year
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['day'] = df['time'].dt.day
        df['month_num'] = df['time'].dt.month
        df['only_date'] = df['time'].dt.date
        df['day_name'] = df['time'].dt.day_name()
        standardtime=[]
        for i, j in df.iterrows():
            if j['am_pm'] == 'pm':
                standardtime.append(j['hour']+12)
            else:
                standardtime.append(j['hour'])
        df['hour'] = standardtime





        period = []
        for hour in df[['day_name', 'hour']]['hour']:
            if hour == 23:
                period.append(str(hour) + "-" + str('00'))
            elif hour == 0:
                period.append(str('00') + "-" + str(hour + 1))
            else:
                period.append(str(hour) + "-" + str(hour + 1))

        df['period'] = period

        xtest=cv.transform(df['messages'])
        df['predict']=mnb.predict(xtest)







    else:
        pattern2 = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

        messages = re.split(pattern2, data)[1:]
        dates = re.findall(pattern2, data)

        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        # convert message_date type
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')

        df.rename(columns={'message_date': 'date'}, inplace=True)

        users = []
        messages = []
        for message in df['user_message']:
            entry = re.split('([\w\W]+?):\s', message)
            if entry[1:]:  # user name
                users.append(entry[1])
                messages.append(" ".join(entry[2:]))
            else:
                users.append('group_notification')
                messages.append(entry[0])

        df['users'] = users
        df['messages'] = messages
        df.drop(columns=['user_message'], inplace=True)

        df['only_date'] = df['date'].dt.date
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        df['day_name'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['messages'].loc[df['messages'] == '<Media omitted>\n'] = '<Media Omitted>'

        period = []
        for hour in df[['day_name', 'hour']]['hour']:
            if hour == 23:
                period.append(str(hour) + "-" + str('00'))
            elif hour == 0:
                period.append(str('00') + "-" + str(hour + 1))
            else:
                period.append(str(hour) + "-" + str(hour + 1))

        df['period'] = period
        xtest=cv.transform(df['messages'])
        df['predict']=mnb.predict(xtest)

    return df
