import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")

import discord
import json
import os
from discord.ext import commands
import asyncio
import youtube_dl
from itertools import cycle
import time

import nltk
#nltk.download()
import random
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import wordnet, movie_reviews

ps = PorterStemmer()
from nltk.corpus import stopwords, state_union
#print(discord.__version__)  # check to make sure at least once you're on the right version!

token = open("token.txt", "r").read()  # I've opted to just save my token to a text file. 

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

### Whats the difference here?
#client = discord.Client()  # starts the discord client.
client = commands.Bot(command_prefix = ".")
#client.remove_command('help')
#status = ['Msg1','Msg2','Msg3']


os.chdir(r'C:\Users\Robin\Desktop\Bot6\bot')
#os.chdir(r'C:\Users\TapperR\Desktop\Bot6\')


#close the bot within server
@client.command()
async def x():
    await client.close()

#### EVENTS ####
@client.event  # event decorator/wrapper. More on decorators here: https://pythonprogramming.net/decorators-intermediate-python-tutorial/
async def on_ready():  # method expected by client. This runs once when connected
    await client.change_presence(game = discord.Game(name = 'Test'))    
    print(f'Bot is ready! We have logged in as {client.user}')  # notification of login.

@client.event
async def on_message(message):
    channel = message.channel

    #each message has a bunch of attributes. Here are a few.
    #check out more by print(dir(message)) for example.
    print(f"{message.channel}: {message.author}: {message.author.name}: {message.content}")

    ### talk talk talk ###
    if "hi there" in message.content.lower() and "Rotaka#6963" in str(message.author):
        author = message.author
        content = message.content
        channel = message.channel
        print(f'A user has send a message')

        await client.send_message(channel, 'hi there')


    #Tokenizing, Stemming and Stopping in Sentences
    if len(message.content) > 50 and len(message.content) < 500 and "Rotaka#6963" in str(message.author):
        EXAMPLE_TEXT = message.content
        stop_words = set(stopwords.words('english'))
        # for i in range(len(sent_tokenize(EXAMPLE_TEXT))):
        #     await client.send_message(channel, sent_tokenize(EXAMPLE_TEXT)[i])
        word_tokens = word_tokenize(EXAMPLE_TEXT)

        # filtered_sentence = []

        # for w in word_tokens:
        #     if w not in stop_words:
        #         filtered_sentence.append(w)

        # await client.send_message(channel, word_tokens)
        # await client.send_message(channel, filtered_sentence)

        for w in word_tokens:
            await client.send_message(channel, ps.stem(w))



    #POS-Tagging
    if len(message.content) > 500 and "Rotaka#6963" in str(message.author):
        try:
            for i in tokenized[:5]:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)
                # chunkGram = r"""Chunk: {<.*>+}
                #                     }<VB.?|IN|DT|TO>+{"""
                # chunkParser = nltk.RegexpParser(chunkGram)
                # chunked = chunkParser.parse(tagged)
                # # chunked.draw()  
                # # await client.send_message(channel, tagged)
                # for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                #     print(subtree)
                namedEnt = nltk.ne_chunk(tagged, binary=False)
                #namedEnt.draw()           

        except Exception as e:
            print(str(e))


    #Synonyms, Antonyms
    if "synonym" in message.content.lower() and "Rotaka#6963" in str(message.author):
        await client.send_message(channel, "Please enter the word: ")
        word = await client.wait_for_message(author=message.author)
        await client.send_message(message.channel, 'You want some synonyms of ' + word.content + ' huh? No problem...')
        await client.send_message(message.channel, 'Give me a moment')

        synonyms = []
        for syno in wordnet.synsets(word.content):
            for l in syno.lemmas():
                if l.name() != word.content:
                    synonyms.append(l.name())
        await client.send_message(message.channel, 'Ok my friend. Heres a list of synonyms:')
        await client.send_message(message.channel, synonyms)


    if "antonym" in message.content.lower() and "Rotaka#6963" in str(message.author):
        await client.send_message(channel, "Please enter the word: ")
        word = await client.wait_for_message(author=message.author)
        await client.send_message(message.channel, 'You want some antonyms of ' + word.content + ' huh? No problem...')
        await client.send_message(message.channel, 'Give me a moment')

        antonyms = []
        for anto in wordnet.synsets(word.content):
            for l in anto.lemmas():
                if l.name() != word.content and l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        await client.send_message(message.channel, 'Ok my friend. Heres a list of antonyms:')
        await client.send_message(message.channel, antonyms)


    if "definition" in message.content.lower() and "Rotaka#6963" in str(message.author):
        await client.send_message(channel, "Please enter the word: ")
        word = await client.wait_for_message(author=message.author)
        await client.send_message(message.channel, 'You want the definition of ' + word.content + ' huh? No problem...')
        await client.send_message(message.channel, 'Give me a moment')
        await client.send_message(message.channel, wordnet.synsets(word.content)[0].definition())
        await client.send_message(message.channel, "An example sentence would be:")
        await client.send_message(message.channel, wordnet.synsets(word.content)[0].examples()[0])


    await client.process_commands(message)



    #Textclassification for movie reviews
    if "film" in message.content.lower() and "Rotaka#6963" in str(message.author):
        documents = [(list(movie_reviews.words(fileid)), category) 
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]

        random.shuffle(documents)
        #await client.send_message(message.channel, documents[0][0][0:20])

        all_words = []
        for w in movie_reviews.words():
            all_words.append(w.lower())

        all_words = nltk.FreqDist(all_words)
        word_features = list(all_words.keys())[:3000]

        def find_features(document):
            words = set(document)
            features = {}
            for w in word_features:
                features[w] = (w in words)

            return features

        #list of tuples
        featuresets = [(find_features(rev), category) for (rev, category) in documents]
        #print(featuresets[2])

        # set that we'll train our classifier with
        training_set = featuresets[:1900]

        # set that we'll test against.
        testing_set = featuresets[1900:]

        #define and train our classifier
        classifier = nltk.NaiveBayesClassifier.train(training_set)

        #test it
        print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)


        #what are the most suspicious words
        classifier.show_most_informative_features(15)



#### COMMANDS ####
@client.command()
async def echo(*args):
    output = ''
    for word in args:
        output += word
        output += ' '
    await client.say(output)


client.run(token) 








#what the bot say if a message was deleted
# @client.event
# async def on_message_delete(message):
#   author = message.author
#   content = message.content
#   channel = message.channel
#   await client.send_message(channel, '{} removed message: {}'.format(author, content))


#### Part 10, Reactions ####
# @client.event
# async def on_reaction_add(reaction, user):
#     channel = reaction.message.channel
#     await client.send_message(channel, '{} has added {} to the message: {}'.format(
#         user.name, reaction.emoji, reaction.message.content))

# async def on_reaction_remove(reaction, user):
#     channel = reaction.message.channel
#     await client.send_message(channel, '{} has removed {} from the message: {}'.format(
#         user.name, reaction.emoji, reaction.message.content))
    





# players = {}
# queues = {}

# def check_queue(id):
# 	if queues[id] != []:
# 		player = queues[id].pop(0)
# 		players[id] = player
# 		player.start()



# @client.command()
# async def ping():
# 	await client.say('Pong!')




#### Part 4, Clear ####
# @client.command(pass_context = True)
# async def clear (ctx, amount = 100):
# 	channel = ctx.message.channel
# 	messages = []
# 	async for message in client.logs_from(channel, limit = int(amount)):
# 		messages.append(message)
# 	await client.delete_messages(messages)
# 	await client.say('Messages deleted')


#### Part 5, Autorole ####
# @client.event
# async def on_member_join(member):
# 	role = discord.utils.get(member.server.roles, name = 'Example Role')
# 	await client.add_roles(member, role)



#### Part 7, Background Tasks ####
# async def change_status():
# 	await client.wait_until_ready()
# 	msgs = cycle(status)

# 	while not client.is_closed:
# 		current_status = next(msgs)
# 		await  client.change_presence(game = discord.Game(name = current_status))
# 		await asyncio.sleep(5)


# client.loop.create_task(change_status())



#### Part 8, Embeds ####

# @client.command(pass_context = True)
# async def displayembed(ctx):
# 	channel = ctx.message.channel
# 	embed = discord.Embed(
# 		title = 'Title',
# 		description = 'This is a description',
# 		colour = discord.Colour.blue()
# 	)

# 	embed.set_footer(text = 'This is a footer')
# 	embed.set_image(url = 'https://cdn.discordapp.com/attachments/536495311984525324/545608874321903631/IMG-20170113-WA0004.jpg')
# 	embed.set_thumbnail(url = 'https://cdn.discordapp.com/attachments/536495311984525324/545608874321903631/IMG-20170113-WA0004.jpg')
# 	embed.set_author(name = 'Author name', 
# 		icon_url = 'https://cdn.discordapp.com/attachments/536495311984525324/545608874321903631/IMG-20170113-WA0004.jpg')
# 	embed.add_field(name = 'Field Name', value = 'Field Value', inline = False)
# 	embed.add_field(name = 'Field Name', value = 'Field Value', inline = True)

# 	#await client.say(embed = embed)
# 	await client.send_message(channel, embed = embed)



# @client.command(pass_context = True)
# async def help(ctx):
# 	author = ctx.message.author

# 	embed = discord.Embed(
# 		colour = discord.Colour.red()
# 	)

# 	embed.set_author(name = 'Help')
# 	embed.add_field(name = '.ping', value = 'Returns Pong!', inline = False)

# 	await client.say('You have a PN')
# 	#sending private message, not in the channel -> author instead of message
# 	await client.send_message(author, embed = embed)





#### Part 11, Join/Leave ####
# @client.command(pass_context = True)
# async def join(ctx):
# 	channel = ctx.message.author.voice.voice_channel
# 	await client.join_voice_channel(channel)


# @client.command(pass_context = True)
# async def leave(ctx):
# 	server = ctx.message.server
# 	voice_client = client.voice_client_in(server)
# 	await voice_client.disconnect()



#### Part 12, Playing Audio ####
# @client.command(pass_context = True)
# async def play(ctx, url):
# 	server = ctx.message.server
# 	#access the voice client of that server
# 	voice_client = client.voice_client_in(server)
# 	player = await voice_client.create_ytdl_player(url, after = lambda:check_queue(server.id))
# 	players[server.id] = player
# 	player.start()



#### Part 13, Pause/Stop/Resume ####

# @client.command(pass_context = True)
# async def pause(ctx):
# 	id = ctx.message.server.id
# 	players[id].pause()


# @client.command(pass_context = True)
# async def stop(ctx):
# 	id = ctx.message.server.id
# 	players[id].stop()


# @client.command(pass_context = True)
# async def resume(ctx):
# 	id = ctx.message.server.id
# 	players[id].resume()


# #### Part 14, Queues ####
# @client.command(pass_context = True)
# async def queue(ctx, url):
# 	server = ctx.message.server
# 	voice_client = client.voice_client_in(server)
# 	player = await voice_client.create_ytdl_player(url)

# 	if server.id in queues:
# 		queues[server.id].append(player)

# 	else:
# 		queues[server.id] = [player]

# 	await client.say('Video queued')


#### Part 15, Level System ####

# @client.event
# async def on_member_join(member):
# 	with open('users.json', 'r') as f:
# 		users = json.load(f)

# 	await update_data(users, member)

# 	with open('users.json', 'w') as f:
# 		json.dump(users, f)



# @client.event
# async def on_message(message):
# 	with open('users.json', 'r') as f:
# 		users = json.load(f)


# 	await update_data(users, message.author)
# 	await add_experience(users, message.author, 5)
# 	await level_up(users, message.author, message.channel)


# 	with open('users.json', 'w') as f:
# 		json.dump(users, f)


# async def update_data(users, user):
# 	if not user.id in users:
# 		users[user.id] = {}
# 		users[user.id]['experience'] = 0
# 		users[user.id]['level'] = 1


# async def add_experience(users, user, exp):
# 	users[user.id]['experience'] += exp

# async def level_up(users, user, channel):
# 	experience = users[user.id]['experience']
# 	lvl_start = users[user.id]['level']
# 	lvl_end = int(experience ** (1/4))

# 	if lvl_start < lvl_end:
# 		await client.send_message(channel, '{} has leveled up to level {}'.format(
# 			user.mention, lvl_end))
# 		users[user.id]['level'] = lvl_end


# #### Part 16, Cogs ####
# extensions = ['fun']

# @client.command()
# async def load(extension):
#     try:
#         client.load_extension(extension)
#         print('Loaded {}'.format(extension))
#     except Exception as error:
#         print('{} cannot be loaded. [{}]'.format(extension, error))

# @client.command()
# async def unload(extension):
#     try:
#         client.load_extension(extension)
#         print('Unloaded {}'.format(extension))
#     except Exception as error:
#         print('{} cannot be unloaded. [{}]'.format(extension, error))


# if __name__ == '__main__':
#     for extension in extensions:
#         try:
#             client.load_extension(extension)
#         except Exception as error:
#             print('{} cannot be loaded. [{}]'.format(extension, error))






# def community_report(guild):
#     online = 0
#     idle = 0
#     offline = 0

#     for m in guild.members:
#         if str(m.status) == "online":
#             online += 1
#         if str(m.status) == "offline":
#             offline += 1
#         else:
#             idle += 1

#     return online, idle, offline


# async def user_metrics_background_task():
# 	await client.wait_until_ready()
# 	global rotaka_guild
# 	rotaka_guild = client.get_guild(536495311984525322)

# 	while not client.is_closed():
# 	    try:
# 	        online, idle, offline = community_report(rotaka_guild)
# 	        with open("usermetrics.csv", "a") as f:
# 	            f.write(f"{int(time.time())},{online},{idle},{offline}\n")
# 	        await asyncio.sleep(5)
# 	        plt.clf()
# 	        df = pd.read_csv("usermetrics.csv", names=['time', 'online', 'idle', 'offline'])
# 	        df['date'] = pd.to_datetime(df['time'],unit='s')
# 	        df['total'] = df['online'] + df['offline'] + df['idle']
# 	        df.drop("time", 1,  inplace=True)
# 	        df.set_index("date", inplace=True)
# 	        df['online'].plot()
# 	        plt.legend()
# 	        plt.savefig("online.png")

# 	        await asyncio.sleep(5)

# 	    except Exception as e:
# 	    	print(str(e))
# 	    	await asyncio.sleep(5) 




# 	if message.content.startswith('$greet'):
# 		await channel.send('Say hello!')

# 		def check(m):
# 			return m.content == 'hello' and m.channel == channel

# 		msg = await client.wait_for('message', check=check)
# 		await channel.send('Hello {.author}!'.format(msg))

# 	if message.content.startswith('$thumb'):
# 		await channel.send('Send me that 👍 reaction, mate')

# 		def check(reaction, user):
# 		    return user == message.author and str(reaction.emoji) == '👍'

# 		try:
# 		    reaction, user = await client.wait_for('reaction_add', timeout=10.0, check=check)
# 		except asyncio.TimeoutError:
# 		    await channel.send('👎')
# 		else:
# 		    await channel.send('👍')


# 	if message.content.startswith('.echo'):
# 		msg = message.content.split()
# 		output = ''
# 		for word in msg[1:]:
# 			output += word
# 			output += " "

# 		await channel.send(output)



# 	if "sample_app.member_count()" == message.content.lower():
# 		await message.channel.send(f"```{rotaka_guild.member_count}```")

# 	elif "sample_app.community_report()" == message.content.lower():
# 		online, idle, offline = community_report(rotaka_guild)
# 		await message.channel.send(f"```Online: {online}.\nIdle/busy/dnd: {idle}.\nOffline: {offline}```")
		

# 		file = discord.File("online.png", filename="online.png")
# 		await message.channel.send("online.png", file=file)


# #what the bot say if a message was deleted
# @client.event
# async def on_message_delete(message):
# 	author = message.author
# 	content = message.content
# 	channel = message.channel
# 	await channel.send('{}: {}: {}'.format(channel, author, content))
