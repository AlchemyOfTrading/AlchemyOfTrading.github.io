#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Tamaghna roy'
SITENAME = 'Alchemy of Quant Trading'
SITEURL = ''

TWITTER_USERNAME="TamaghRoy"

PATH = 'content'

TIMEZONE = 'Asia/Hong_Kong'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('PortfolioCharts', 'https://portfoliocharts.com/'),
         ('AllocateSmartly', 'https://allocatesmartly.com/'),
         ('EconomicPicData', 'http://econompicdata.blogspot.hk/'),
         ('Flirting with Models ', 'https://blog.thinknewfound.com/'),
         ('Investment Idiocy', 'https://qoppac.blogspot.hk/'),
         ('Quantocracy', 'https://qoppac.blogspot.hk/'))

# Social widget
SOCIAL = (('linkedin', 'https://www.linkedin.com/in/tamaghna-roy-89b0696/'),
          ('facebook', 'https://www.facebook.com/tamaghna.roy'),
          ('twitter', 'https://twitter.com/TamaghRoy'),
          ('google-plus', 'https://plus.google.com/+TamaghnaRoy1984'),
          ('github', 'https://github.com/tamaghnaroy'))

DEFAULT_PAGINATION = 10
# IPYNB_USE_META_SUMMARY = True

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
MARKUP = ('md', 'ipynb')

THEME = "themes/tuxlite_tbs"

PLUGIN_PATHS = ['./plugins']
PLUGINS = ['ipynb.markup']

# TYPOGRIFY = False
YEAR_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = 'posts/{date:%Y}/{date:%b}/index.html'

