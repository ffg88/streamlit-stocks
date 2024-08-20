import datetime
from dotenv import load_dotenv

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults


# Loading API KEY from .env and defining the LLM model
load_dotenv()
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Setting current date
curr_date = str(datetime.date.today())
last_year = str(datetime.date.today() - datetime.timedelta(weeks=52))

# Creating Tool to search stock prices for a ticket
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start=last_year, end=curr_date)
    return stock

yahoo_finance_tool = Tool(
    name = 'Yahoo Finance Tool',
    description = 'Fetches stock prices for {ticket} from a specified date from Yahoo Finance API',
    func = lambda ticket: fetch_stock_price(ticket)
)

# Creating the Agent who will analyse the stock price
stockPriceAnalyst = Agent(
    role = 'Senior Stock Price Analyst',
    goal = 'Find the {ticket} stock price and analyses trends',
    backstory = 'You have extensive experience in analysing the price of a specified stock and making predictions about its future price.',
    llm = llm,
    verbose = True,
    max_iter = 5,
    memory = True,
    allow_delegation = False,
    tools = [yahoo_finance_tool]
)

# Creating stock price Task for the Agent
getStockPrice = Task(
    description = 'Analyse the {ticket} stock price history and create a trend analysis of up, down or sideways',
    expected_output = '''Specify the current trend stock price - up, down or sideways.
    eg. stock='AAPL, price up' ''',
    agent = stockPriceAnalyst
)

# Using the DuckDuckGo Tool to search news for a stock ticket company
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# Creating the Agent who will analyse the stock company news
newsAnalyst = Agent(
    role = 'Stock News Analyst',
    goal = '''Create a short summary of the market news related to the stock {ticket} company. Specify the
    current trend - up, down or sideways with the news context. For each request stock asset, specify a number between 0 and 100,
    where 0 is extreme fear and 100 is extreme greed.''',
    backstory = '''You're highly experienced in analysing the market trends and news and have trecked assets for more than 10 year.
    You're also a master level analyst in traditional markets and have deep understanding in human psychology.
    You understand news, their titles and information, but you look at those with a healthy dose of skepticism.
    You consider also the source of the news articles.''',
    llm = llm,
    verbose = True,
    max_iter = 10,
    memory = True,
    allow_delegation = False,
    tools = [search_tool]
)

# Creating stock news Task for the Agent
getNews = Task(
    description = f'''Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.
    The current date is {curr_date}.
    Compose the results into a helpful report.''',
    expected_output = '''A summary of the overall market and a one sentence summary for each requested asset.
    Include a fear/greed score for each asset based on the news. Use the format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>''',
    agent = newsAnalyst
    )

# Creating the Agent who will analyse the stock price
stockAnalystWriter = Agent(
    role = 'Senior Stock Analyst Writer',
    goal = '''Analyse the trends price and news and write an insightful and informative 3 paragraphs long newsletter based on
    the stock report and price trend.''',
    backstory = '''You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling
    stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses.
    You're able to hold multiple opinions when analysing anything.''',
    llm = llm,
    verbose = True,
    max_iter = 5,
    memory = True,
    allow_delegation = True,
)

# Creating stock news Task for the Agent
writeAnalysis = Task(
    description = '''Use the stock price trend and the stock news report to create an analysis and write the newsletter
    about the {ticket} company that is brief and highlight the most import points.
    Focus in the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analysis of stock trend and news summary.''',
    expected_output = '''An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - Main part provides the meat of the analysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction - up, down or sideways.''',
    agent = stockAnalystWriter,
    context = [getStockPrice, getNews]
    )

# Creating Crew
crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks = [getStockPrice, getNews, writeAnalysis],
    verbose = True,
    process = Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)

results = crew.kickoff(inputs={'ticket': 'AAPL'})
