import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessor,helper




st.sidebar.title('Whatsapp Chat Analyser')
uploaded_file = st.sidebar.file_uploader('choose a whatsapp file')
if uploaded_file is not None:
    data=uploaded_file.getvalue()
    data=data.decode('utf-8')
    df=preprocessor.preprocess(data)

    userlist=df['users'].unique().tolist()
    userlist=sorted(userlist)
    userlist.insert(0,'Overall')
    sentiment=df['predict'].unique().tolist()
    sentiment.insert(0,'Overall')

    selected_user=st.sidebar.selectbox('Show Analysis WRT',userlist)
    selected_sentiment=st.sidebar.selectbox('Show Analysis Sentiment',sentiment)



    if st.sidebar.button('Show Analysis'):
        num_messages,words,num_media_messages,num_links=helper.fetch_stats(selected_user,df,selected_sentiment)

        st.title('Top Statistics')
        col1,col2,col3,col4=st.columns(4)

        with col1:
            st.header(" Messages")
            st.title(num_messages)
        with col2:
            st.header("Words")
            st.title(words)
        with col3:
            st.header("Media")
            st.title(num_media_messages)
        with col4:
            st.header("links")
            st.title(num_links)


        if selected_user=='Overall' and selected_sentiment=='Overall':
            st.title('Overall Sentiment Percentage')
            cate=df.groupby('predict')['messages'].count().reset_index()
            fig,ax=plt.subplots()
            ax.pie(cate['messages'],labels=cate['predict'],autopct='%0.2f%%')
            st.pyplot(fig)

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df,selected_sentiment)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['messages'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df,selected_sentiment)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['messages'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df,selected_sentiment)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df,selected_sentiment)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df,selected_sentiment)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df,selected_sentiment)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)


            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df,selected_sentiment)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df,selected_sentiment)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        st.title('Frequently Used Emoji')
        emoji_df=helper.emoji_helper(selected_user,df,selected_sentiment)
        fig,ax=plt.subplots()
        ax.bar(emoji_df[0],emoji_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
