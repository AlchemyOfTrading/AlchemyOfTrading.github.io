from plugins.ipynb.core import get_html_from_filepath
from plugins.ipynb.markup import MyHTMLParser, IPythonNB

reader = IPythonNB(settings=dict(READER=dict(), MD_EXTENSIONS=[], FORMATTED_FIELDS=[], SUMMARY_MAX_LENGTH=None))
content, metadata = reader.read(filepath='../content/Stochastic_Simulation.ipynb')

print(metadata['summary'])
