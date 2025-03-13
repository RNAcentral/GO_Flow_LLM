import guidance
from guidance import user, assistant, select, substring

import logging

logger = logging.getLogger(__name__)


@guidance
def extract_evidence(llm, article_text, mode="recursive-paragraph"):
    """
    Choose some evidence from the article text to support
    the claim just made.

    Mode choices are:
    - recursive-paragraph: Split the article text into paragraphs,
        select a paragraph as most relevant, then use
        substring on just that paragraph
    - recursive-sentence: Split the article into sentences,
        then apply recursive select to pick a few sentences.
    - single-paragraph: Split into paragraphs, select one and
        return that
    - single-sentence: Split into sentences, select one and
        return that
    - full-substring: Run substring selection on the whole
        article text
    """
    logger.info(f"Extracting evidence from article text")
    with user():
        llm += "Give a piece of evidence from the text that supports your answer. "
    if mode == "full-substring":
        with user():
            llm += "Choose the most relevant sentence or two.\n"
        with assistant():
            llm += f"The most relevant piece of evidence is: '{substring(article_text, name='evidence')}'"
    elif mode == "single-sentence":
        # first split the text into sentences by splitting on '. '
        # NB, may not be 100% accurate, but will probably do
        article_sentences = article_text.split(". ")
        with user():
            llm += "Choose the most relevant sentence from the article\n"
        with assistant():
            llm += "The most relevant sentence is: " + select(
                article_sentences, name="evidence"
            )
    elif mode == "single-paragraph":
        # first split the text into paragraphs by splitting on '\n'
        # TODO - check this actually produces paragraph shaped things
        article_paragraphs = list(
            filter(lambda x: len(x) > 0, article_text.split("\n"))
        )
        with user():
            llm += "Choose the most relevant paragraph from the article\n"
        with assistant():
            llm += "The most relevant paragraph is: " + select(
                article_paragraphs, name="evidence"
            )
    elif mode == "recursive-paragraph":
        # first split the text into paragraphs by splitting on '\n'
        # TODO - check this actually produces paragraph shaped things
        article_paragraphs = list(
            filter(lambda x: len(x) > 0, article_text.split("\n"))
        )
        with user():
            llm += "Choose the most relevant paragraph from the article\n"
        with assistant():
            llm += f"The most relevant paragraph is: {select(article_paragraphs, name='relevant_para')}\n"
        paragraph = llm["relevant_para"]
        logger.debug(f"chose this paragraph:\n{paragraph}\n")
        with user():
            llm += "Now choose the most relevant piece of evidence within that paragraph.\n"
        with assistant():
            llm += f"The most relevant piece of evidence is: '{substring(paragraph, name='evidence')}'"
    elif mode == "recursive-sentence":
        # first split the text into sentences by splitting on '. '
        article_sentences = article_text.split(". ")
        with user():
            llm += "Choose the most relevant sentences from the article\n"
        with assistant():
            llm += f"The most relevant sentences are: {select(article_sentences, name='evidence', recurse=True, list_append=True)}\n"
    logger.debug(f"chosen evidence snippet: {llm['evidence']}")
    return llm
