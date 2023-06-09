# https://platform.openai.com/tokenizer

# 138 tokens - prompt
# 3097 - context
# 16 - question
# 800 tokens - response
4097


# V3
"""\n
You are a funny sports journalist writing an article based on a prompt.  Write in the style of Bill Burr and use two curse words.  Use the context below to answer the question.  Use this format, replacing text in brackets with the result.  Do not inclued the brackets in the output:\n\n

Artilce:\n
[Introductory paragraph]\n\n

# [Name of Topic 1]\n
[Paragraph about topic 1]\n\n

[Concluding paragraph]\n\n

Context:\n\n

"""\n\n

Question:  Should the New England Patriots attempt to sign Aaron Rodgers to be their QB?

"""\nYou are a funny sports journalist writing an article based on a prompt.  Write in the style of Bill Burr and use two curse words.  Use the context below to answer the question.  Use this format, replacing text in brackets with the result.  Do not inclued the brackets in the output:\n\nArtilce:\n[Introductory paragraph]\n\n# [Name of Topic 1]\n[Paragraph about topic 1]\n\n[Concluding paragraph]\n\nContext:\n\n{context}"""\n\nQuestion: {question}?\n


# V2

% System\n\n

You are a funny sports journalist.  You write humerous articles about trending football news.  You write in the style of Bill Burr, use foul language, and close all of your articles with a witty tagline.  All of your articles have 4 paragraphs as follows:\n\n

1. Introduction\n
2. Supporting point 1\n
3. Supporting point 2\n
4. Conclusion\n\n

Read and apply the following examples when responding to questions.\n\n

% Context\n\n

Example 1:\n
Context and details.\n\n

Example 2:\n
Context and details.\n\n

----\n\n

% Question\n
% \n\n

% Answer\n
%


% System\n\nYou are a funny sports journalist.  You write humerous articles about trending football news.  You write in the style of {author}, use foul language, and close all of your articles with a witty tagline.  All of your articles have 4 paragraphs as follows:\n\n1. Introduction\n2. Supporting point 1\n3. Supporting point 2\n4. Conclusion\n\nRead and apply the following examples when responding to questions.\n\n% Context\n\n{context}\n\n----\n\n% Question\n% {question}\n\n% Answer\n%




# 184 tokens - prompt
# 3097 - context
# 16 - question
# 800 tokens - response
4097

# V1

“””
You are a funny sports journalist writing an article based on a prompt.  Use the following format, replacing text in brackets with the result.  Write in the style of Jerry Seinfeld and answer the question based upon the context below.  If the question can’t be answered based on the context, say \”I don’t know \”\n\n

Context: {context}\n

Question: {question}\n

Answer: 

[[introductory paragraph]]\n

## [name of topic 1]
[[paragraph about topic 1]]\n

## [name of topic 2]
[[paragraph about topic 2]]\n

[[concluding paragraph]]
“””

### Wrapped

'“””\nYou are a funny sports journalist writing an article based on a prompt.  Use the following format, replacing text in brackets with the result.  Write in the style of Jerry Seinfeld and answer the question based upon the context below.  If the question can’t be answered based on the context, say \”I don’t know \”\n\nContext: {context}\nQuestion: {question}\nAnswer: [[introductory paragraph]]\n## [name of topic 1] [[paragraph about topic 1]]\n## [name of topic 2] [[paragraph about topic 2]]\n## [[concluding paragraph]]“””'










# Questions

print(answer_question(df, question="Should the New England Patriots attempt to sign Aaron Rodgers to be their QB?"))

print(answer_question(df, question="Who was the most valuable player on the New England Patriots during the 2022 season?"))

print(answer_question(df, question="Should the New England Patriots trade for Baltimore Ravens QB Lamar Jackson?"))



# Answer:

I have just one question for the New England Patriots: What took you so long to realize that if you wanted to contend, it was going time to sign a real QB? Sure, Mac Jones had his moments this past season, but he's not an elite gunslinger. On the other side of things, Aaron Rodgers is as good as they come! If I were Bill Belichick, I would've thrown my hat in the ring and brought in a Hall-of-Famer who can carry me back into the playoffs. 

Up until now, all we ever hear about is Tom Brady and how he won six Super Bowls in New England. Everyone seems convinced that any quarterback can replicate what Brady did here - well newsflash: they can't! It'd be different if Brees or Mahomes signed with them; there's no denying their greatness and ability to lead a team through anything. But no matter your opinion on signing Aaron Rodgers for potentially big money, it’d certainly energize fans in New England…and give old man Belitck another shot at hoisting up Lombardi Trophy number seven! 

The writing appears finally on the wall - could Aaron Rodgers be donning red and silver next year? Maybe...This move makes too much sense not to happen. The organization should do what most smart teams do when they run out of options – go get an established star player who knows how championships are won in crunchtime situations; who else better fits that profile than A-Rod?!


No chance the Patriots should attempt to sign Aaron Rodgers. They already have a good quarterback in Mac Jones, who they just drafted in April. He was one of the best QB prospects and has proven he can operate at a high level from clean pockets. Plus, they have Bailey Zappe who could push Jones next season and provide some invaluable competition. It's time for the Pats to start investing their resources into other positions like wide receiver where there is an urgent need for improvement if New England wants to become competitive again in 2023. Also don't forget about Brian Hoyer: with his veteran experience, he might be worth that cap number as it looks like he can still work his magic on current active player contract if not ready to accept less money for more hours under a coaching role quite yet. 

More importantly though - let us look at two players who had outstanding seasons so far: Dont'a Hightower and J..J Taylor, both of whom will be key pieces going forward into 2023 . Hightower is said to anchor the team’s defense this coming year; whereas Taylor is building up some buzz even amongst all those running backs I mentioned earlier (Damien Harris , Rex Burkhead etc). Then Brandon Bolden also opted back in which shows promise looking toward 2021 season altogether! 

So no matter what anyone else thinks - these guys got something going here already - whether its by fostering old talent or infusing young blood. All considering why invest so much time energy (not mention money) chasing after Aaron Rodgers when you might miss out on maximizing what you already have? Especially since Cam Newton proved last year that sometimes age IS just a number ;)  



# Context

Quick-Hit Thoughts on Every Player on the Patriots Roster During the 2022 Season///The Patriots season came to a disappointing end in Sunday's loss to the Bills, where New England would've earned a trip to the postseason with a win in Orchard Park.Unfortunately, the Week 18 matchup didn't go as planned, so the Patriots are heading into an early offseason that should be filled with significant organizational changes to get back to the playoffs.Here's a note compiled from our weekly film reviews on every single player currently on the Patriots roster who played a snap in 2022:QuarterbackMac Jones - We wrote an extensive breakdown of the starting quarterback here. Mac has proven to be a functional NFL starter capable of operating at a high level from clean pockets. However, it remains to be seen how much he can elevate an NFL offense, especially when he's going toe-to-toe with an elite quarterback. To date, he's still searching for a signature win against a playoff team at full strength. Next season, the Pats should level the playing field with an experienced OC and a top weapon at Jones's disposal.Bailey Zappe (rookie) - The 2021 fourth-round pick made the season interesting when he filled in for nearly three games when Jones was injured. Zappe showed poise, clean mechanics to produce accurate throws, and impressive processing speed for a first-year QB. He proved himself as a fringe starter/high-end backup who could push Jones next summer. However, Zappe's physical limitations to create plays on his own and under pressure are also concerns.Brian Hoyer - Hoyer is under contract for the 2023 season at a $2.24 million cap hit. With Zappe as the backup, it's fair to wonder if Hoyer is worth that cap number. But, even on injured reserve, he was a constant at the facility to the point where it's fair to assume he was a de facto coach. Would he surrender his roster spot for a permanent coaching role? Probably not. Why take less money to work more hours when you can do the job on an active player contract?The Patriots season came to a disappointing end in Sunday's loss to the Bills, where New England would've earned a trip to the postseason with a win in Orchard Park.

###

2022 Breakout Patriots to build around///The Patriots finished the 2022 season at 8-9 and missed the playoffs for the second time in three years, yet young players continued to emerge, players who should form a new core in New England that will one day lead the team back to postseason success. With three-straight drafts filled with immediate contributors, the once-heavily-veteran squad has slowly tipped the scales toward a youth movement whose development will be a critical part of 2023.Here's one rookie, two second-years and two third-years that put together outstanding seasons and should be reasons for optimism moving forward and long-term building blocks, some of whom could be extended as soon as this offseason.The Patriots finished the 2022 season at 8-9 and missed the playoffs for the second time in three years, yet young players continued to emerge, players who should form a new core in New England that will one day lead the team back to postseason success. With three-straight drafts filled with immediate contributors, the once-heavily-veteran squad has slowly tipped the scales toward a youth movement whose development will be a critical part of 2023.

###

Biletnikoff Award Winner Will Finally Make His Patriots Debut in 2021///Getty                                                                   Marqise Lee                                                                                      After being hit the hardest of any team in the NFL by COVID-19 opt-outs in 2020, the New England Patriots appear set to get back at least some of the pieces it was missing this season.The Patriots had an NFL-high 8 players opt-out of the 2020 season due to concerns about COVID-19. At least 3 and possibly 4 of the players who opted out were sorely missed as the Patriots missed the postseason in 2020 for the first time since 2008.Thankfully, it appears at least some of the players who opted out are planning to return in 2021. Follow the Heavy on Patriots Facebook page for the latest breaking news, rumors, and content!Marqise LeeMuch has been said about the Patriots’ void at wide receiver.While he’s not a star, we didn’t hear many references to Marqise Lee, a player the Patriots signed to a one-year prove-it deal during the offseason. No one expected Lee to turn into Marvin Harrison in 2020, but can anyone definitively say he wouldn’t have been arguably the team’s No. 1 receiver had he not opted out?We’ll never know the answer to that question, but perhaps we’ll see some indication one way or the other in 2021 as it appears he’s planning to return. According to ESPN’s Mike Reiss, Lee said, “I got to Boston on Sunday, did the physical and everything was perfect. As far as this year goes, I’m eager to get back.”Lee said he had no regrets after opting out as he had a baby daughter on the way, but is said to be “leaning toward” returning in 2021.Brandon BoldenRunning back and special teams ace Brandon Bolden has already said he plans to return in 2021.The Patriots have a wealth of running backs, but there are some injury and aging situations the team has to navigate. Rex Burkhead was lost for the season with a torn ACL. Damien Harris had 2 stints on injured reserve–including to end the season–and James White is a free agent who will be 29 in February.

###

I mean, they did draft Mac Jones in the first round in April. But Cam knows what time it is. He even acknowledged it in a recent interview with ESPN Radio."For me, the Patriots' organization has been impeccable," Newton said. "So my time there has been everything I could have asked for. I guess it's now time for me to uphold my end of the bargain, through and through."Cam averaged 177.1 passing yards per game in 2020 (fewest in the NFL among 35 qualified quarterbacks). He also ranked toward the bottom of that group in TD-to-INT ratio (8:10) and passer rating (82.9). But he had 592 rushing yards (third most among QBs) and 12 rushing touchdowns (most among QBs) last year. So, I'll say it: He still can play at an elite level. Some might think the 32-year-old is a bit too old to do it. But you know who else they said that about? C.T. from the MTV's The Challenge. And guess who won The Challenge: Double Agents last season? That's right, Boston's own CT. So show some respect for your elders.Projected 2021 MVP: Dont'a Hightower, linebacker. The Patriots are one of those teams that seemingly make a surprise player cut coming out of camp every year, and we act like it's going to be the downfall of the program. Yet, they still manage to move on. Last year was different. It seemed like the Patriots didn't get over losing Hightower, who opted out of the 2020 season. New England ranked 15th in yards allowed last season. With Hightower coming back to anchor the defense, perhaps the Patriots' D will get back into the top 10 in 2021.2021 breakout star: J.J. Taylor, running back. The Patriots have a lot of running backs on their roster. Damien Harris is the guy who gets talked about the most (at least in fantasy football terms). Sony Michel is a former first-round pick. And let's not forget about James White. There is a little buzz building for Taylor, though. Remember, even with so many backs in the mix, the Patriots always found a way to get the ball to Rex Burkhead, at least when he was healthy.

###

Patriots Veteran Ranked as Team’s No. 1 Player///Getty                                                                                          Matthew Judon                                                                                    Is quarterback Mac Jones the New England Patriots’ No. 1 player? Not according to WEEI’s Andy Hart.In a Patriots player-ranking piece listing each guy from best to worst, Jones finished seventh on the list. Who was No. 1? Veteran pass rusher Matthew Judon. In a brief explanation for Judon’s lofty spot on the list, Hart wrote:“He was a Pro Bowler in Baltimore and he’s a Pro Bowler in New England.” There are no lies detected in Hart’s statement. Judon made the Pro Bowl in 2021 after his first season in New England. The 29-year-old made his third straight Pro Bowl and set a career-high with 12.5 sacks.His 14 approximate value rating was also the highest in his NFL career. Judon is again expected to power the Patriots’ defense in 2022, though there figures to be a number of new faces on the unit with him.As of now, it appears as though Dont’a Hightower will not be returning. He could be replaced by Raekwon McMillan or even Cameron McGrone at middle linebacker. The Patriots brought back Jamie Collins in 2021, but he too isn’t on the Patriots roster currently.New England traded Chase Winovich to the Cleveland Browns for Mack Wilson, who will also figure into the linebacker situation. While more athletic, the Patriots’ linebacker corps will be younger and less experienced.How that affects the overall quality of the unit remains to be seen.ALL the latest Patriots news straight to your inbox! Join the Heavy on Patriots newsletter here! Join Heavy on Patriots! The other players ranked ahead of Jones in Hart’s ranking were Nick Folk, Adrian Phillips, Kyle Dugger, Christian Barrmore and Trent Brown.New England Has 2 of the Best-Value Contracts in the NFLJudon isn’t just one of the Patriots’ best players, he is perhaps the biggest bargain. According to Pro Football Focus, Judon and Patriots wide receiver Kendrick Bourne have the 20th and 17th best-valued contracts in the NFL.Judon produced in a major way for the Patriots in 2021, and he won’t be 30 until August. This suggests he still has enough time to remain productive throughout the life of his four year, $54.5 million deal.

###

Patriots Named One of Biggest Losers of 2022 NFL Offseason///Getty                                                                              New England Patriots                                                                                     While the New England Patriots have made some moves this offseason, some experts still think they lost out.ALL the latest Patriots news straight to your inbox! Join the Heavy on Patriots newsletter here! The Patriots are a year removed from an offseason where they spent an exorbitant amount. In 2021, New England brought in Jalen Mills, Nelson Agholor, Hunter Henry, Jonnu Smith, and Matthew Judon last offseason. While some of those moves were a success (Judon and Henry) some players struggled (Agholor and Smith).Bleacher Report’s Alex Ballentine took a look at the winners and losers of the 2022 offseason and while the rest of the AFC East improved, the same can’t be said for New England according to the analyst.“The Jets made this list as a winner,” Ballentine stated. “So did McDaniel and, by extension, the Dolphins. The Buffalo Bills are once again Super Bowl favorites. That leaves the Patriots as the only team within the division that can’t say it got appreciably better.”Ballentine added that while New England is a loser, Mac Jones could lift the Patriots into a playoff spot.“Fortunately, Mac Jones showed during his rookie year that he has the potential to be a franchise quarterback,” Ballentine continued. “His development could keep the Patriots in the thick of things within the division. But nothing the team did this offseason moves the needle.”Who Has New England Gained This Offseason?New England’s biggest move is without a doubt DeVante Parker. The wide receiver is now the only one on the Patriots roster to have ever recorded a 1,000 receiving yard season. While Parker struggled to stay healthy last season, New England will be looking to revitalize his career.Mack Wilson will be looking to do the same after he was traded to the Patriots in exchange for Chase Winovich. Both parts of the trade struggled with their now former teams and will be excited for a fresh start in 2022.New England also brought back Malcolm Butler.


###################################################################
Context - 3116

Quick-Hit Thoughts on Every Player on the Patriots Roster During the 2022 Season///The Patriots season came to a disappointing end in Sunday's loss to the Bills, where New England would've earned a trip to the postseason with a win in Orchard Park.Unfortunately, the Week 18 matchup didn't go as planned, so the Patriots are heading into an early offseason that should be filled with significant organizational changes to get back to the playoffs.Here's a note compiled from our weekly film reviews on every single player currently on the Patriots roster who played a snap in 2022:QuarterbackMac Jones - We wrote an extensive breakdown of the starting quarterback here. Mac has proven to be a functional NFL starter capable of operating at a high level from clean pockets. However, it remains to be seen how much he can elevate an NFL offense, especially when he's going toe-to-toe with an elite quarterback. To date, he's still searching for a signature win against a playoff team at full strength. Next season, the Pats should level the playing field with an experienced OC and a top weapon at Jones's disposal.Bailey Zappe (rookie) - The 2021 fourth-round pick made the season interesting when he filled in for nearly three games when Jones was injured. Zappe showed poise, clean mechanics to produce accurate throws, and impressive processing speed for a first-year QB. He proved himself as a fringe starter/high-end backup who could push Jones next summer. However, Zappe's physical limitations to create plays on his own and under pressure are also concerns.Brian Hoyer - Hoyer is under contract for the 2023 season at a $2.24 million cap hit. With Zappe as the backup, it's fair to wonder if Hoyer is worth that cap number. But, even on injured reserve, he was a constant at the facility to the point where it's fair to assume he was a de facto coach. Would he surrender his roster spot for a permanent coaching role? Probably not. Why take less money to work more hours when you can do the job on an active player contract?The Patriots season came to a disappointing end in Sunday's loss to the Bills, where New England would've earned a trip to the postseason with a win in Orchard Park.

###

2022 Breakout Patriots to build around///The Patriots finished the 2022 season at 8-9 and missed the playoffs for the second time in three years, yet young players continued to emerge, players who should form a new core in New England that will one day lead the team back to postseason success. With three-straight drafts filled with immediate contributors, the once-heavily-veteran squad has slowly tipped the scales toward a youth movement whose development will be a critical part of 2023.Here's one rookie, two second-years and two third-years that put together outstanding seasons and should be reasons for optimism moving forward and long-term building blocks, some of whom could be extended as soon as this offseason.The Patriots finished the 2022 season at 8-9 and missed the playoffs for the second time in three years, yet young players continued to emerge, players who should form a new core in New England that will one day lead the team back to postseason success. With three-straight drafts filled with immediate contributors, the once-heavily-veteran squad has slowly tipped the scales toward a youth movement whose development will be a critical part of 2023.

###

Biletnikoff Award Winner Will Finally Make His Patriots Debut in 2021///Getty                                                                   Marqise Lee                                                                                      After being hit the hardest of any team in the NFL by COVID-19 opt-outs in 2020, the New England Patriots appear set to get back at least some of the pieces it was missing this season.The Patriots had an NFL-high 8 players opt-out of the 2020 season due to concerns about COVID-19. At least 3 and possibly 4 of the players who opted out were sorely missed as the Patriots missed the postseason in 2020 for the first time since 2008.Thankfully, it appears at least some of the players who opted out are planning to return in 2021. Follow the Heavy on Patriots Facebook page for the latest breaking news, rumors, and content!Marqise LeeMuch has been said about the Patriots’ void at wide receiver.While he’s not a star, we didn’t hear many references to Marqise Lee, a player the Patriots signed to a one-year prove-it deal during the offseason. No one expected Lee to turn into Marvin Harrison in 2020, but can anyone definitively say he wouldn’t have been arguably the team’s No. 1 receiver had he not opted out?We’ll never know the answer to that question, but perhaps we’ll see some indication one way or the other in 2021 as it appears he’s planning to return. According to ESPN’s Mike Reiss, Lee said, “I got to Boston on Sunday, did the physical and everything was perfect. As far as this year goes, I’m eager to get back.”Lee said he had no regrets after opting out as he had a baby daughter on the way, but is said to be “leaning toward” returning in 2021.Brandon BoldenRunning back and special teams ace Brandon Bolden has already said he plans to return in 2021.The Patriots have a wealth of running backs, but there are some injury and aging situations the team has to navigate. Rex Burkhead was lost for the season with a torn ACL. Damien Harris had 2 stints on injured reserve–including to end the season–and James White is a free agent who will be 29 in February.

###

I mean, they did draft Mac Jones in the first round in April. But Cam knows what time it is. He even acknowledged it in a recent interview with ESPN Radio."For me, the Patriots' organization has been impeccable," Newton said. "So my time there has been everything I could have asked for. I guess it's now time for me to uphold my end of the bargain, through and through."Cam averaged 177.1 passing yards per game in 2020 (fewest in the NFL among 35 qualified quarterbacks). He also ranked toward the bottom of that group in TD-to-INT ratio (8:10) and passer rating (82.9). But he had 592 rushing yards (third most among QBs) and 12 rushing touchdowns (most among QBs) last year. So, I'll say it: He still can play at an elite level. Some might think the 32-year-old is a bit too old to do it. But you know who else they said that about? C.T. from the MTV's The Challenge. And guess who won The Challenge: Double Agents last season? That's right, Boston's own CT. So show some respect for your elders.Projected 2021 MVP: Dont'a Hightower, linebacker. The Patriots are one of those teams that seemingly make a surprise player cut coming out of camp every year, and we act like it's going to be the downfall of the program. Yet, they still manage to move on. Last year was different. It seemed like the Patriots didn't get over losing Hightower, who opted out of the 2020 season. New England ranked 15th in yards allowed last season. With Hightower coming back to anchor the defense, perhaps the Patriots' D will get back into the top 10 in 2021.2021 breakout star: J.J. Taylor, running back. The Patriots have a lot of running backs on their roster. Damien Harris is the guy who gets talked about the most (at least in fantasy football terms). Sony Michel is a former first-round pick. And let's not forget about James White. There is a little buzz building for Taylor, though. Remember, even with so many backs in the mix, the Patriots always found a way to get the ball to Rex Burkhead, at least when he was healthy.

###

Patriots Veteran Ranked as Team’s No. 1 Player///Getty                                                                                          Matthew Judon                                                                                    Is quarterback Mac Jones the New England Patriots’ No. 1 player? Not according to WEEI’s Andy Hart.In a Patriots player-ranking piece listing each guy from best to worst, Jones finished seventh on the list. Who was No. 1? Veteran pass rusher Matthew Judon. In a brief explanation for Judon’s lofty spot on the list, Hart wrote:“He was a Pro Bowler in Baltimore and he’s a Pro Bowler in New England.” There are no lies detected in Hart’s statement. Judon made the Pro Bowl in 2021 after his first season in New England. The 29-year-old made his third straight Pro Bowl and set a career-high with 12.5 sacks.His 14 approximate value rating was also the highest in his NFL career. Judon is again expected to power the Patriots’ defense in 2022, though there figures to be a number of new faces on the unit with him.As of now, it appears as though Dont’a Hightower will not be returning. He could be replaced by Raekwon McMillan or even Cameron McGrone at middle linebacker. The Patriots brought back Jamie Collins in 2021, but he too isn’t on the Patriots roster currently.New England traded Chase Winovich to the Cleveland Browns for Mack Wilson, who will also figure into the linebacker situation. While more athletic, the Patriots’ linebacker corps will be younger and less experienced.How that affects the overall quality of the unit remains to be seen.ALL the latest Patriots news straight to your inbox! Join the Heavy on Patriots newsletter here! Join Heavy on Patriots! The other players ranked ahead of Jones in Hart’s ranking were Nick Folk, Adrian Phillips, Kyle Dugger, Christian Barrmore and Trent Brown.New England Has 2 of the Best-Value Contracts in the NFLJudon isn’t just one of the Patriots’ best players, he is perhaps the biggest bargain. According to Pro Football Focus, Judon and Patriots wide receiver Kendrick Bourne have the 20th and 17th best-valued contracts in the NFL.Judon produced in a major way for the Patriots in 2021, and he won’t be 30 until August. This suggests he still has enough time to remain productive throughout the life of his four year, $54.5 million deal.

###

Patriots Named One of Biggest Losers of 2022 NFL Offseason///Getty                                                                              New England Patriots                                                                                     While the New England Patriots have made some moves this offseason, some experts still think they lost out.ALL the latest Patriots news straight to your inbox! Join the Heavy on Patriots newsletter here! The Patriots are a year removed from an offseason where they spent an exorbitant amount. In 2021, New England brought in Jalen Mills, Nelson Agholor, Hunter Henry, Jonnu Smith, and Matthew Judon last offseason. While some of those moves were a success (Judon and Henry) some players struggled (Agholor and Smith).Bleacher Report’s Alex Ballentine took a look at the winners and losers of the 2022 offseason and while the rest of the AFC East improved, the same can’t be said for New England according to the analyst.“The Jets made this list as a winner,” Ballentine stated. “So did McDaniel and, by extension, the Dolphins. The Buffalo Bills are once again Super Bowl favorites. That leaves the Patriots as the only team within the division that can’t say it got appreciably better.”Ballentine added that while New England is a loser, Mac Jones could lift the Patriots into a playoff spot.“Fortunately, Mac Jones showed during his rookie year that he has the potential to be a franchise quarterback,” Ballentine continued. “His development could keep the Patriots in the thick of things within the division. But nothing the team did this offseason moves the needle.”Who Has New England Gained This Offseason?New England’s biggest move is without a doubt DeVante Parker. The wide receiver is now the only one on the Patriots roster to have ever recorded a 1,000 receiving yard season. While Parker struggled to stay healthy last season, New England will be looking to revitalize his career.Mack Wilson will be looking to do the same after he was traded to the Patriots in exchange for Chase Winovich. Both parts of the trade struggled with their now former teams and will be excited for a fresh start in 2022.New England also brought back Malcolm Butler.



"""
I want you to act as Bill Burr the stand-up comedian. Below I've provided you with some examples related to current events and you will use your wit, creativity, and observational skills to create a routine based on those examples. You should also be sure to incorporate personal anecdotes or experiences into the routine in order to make it more relatable and engaging for the audience. 

Context: