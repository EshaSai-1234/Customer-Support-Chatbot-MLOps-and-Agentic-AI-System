"""
Tests for AI Chatbot Tools and Conversation Engine
"""

import pytest
from src.chatbot.tools import ComplaintChatbotTools
from src.chatbot.bot import ComplaintIntelligenceBot


def test_chatbot_tools_overview():
    stats = ComplaintChatbotTools.get_overview_statistics()
    assert "total_complaints" in stats
    assert stats["total_complaints"] > 0
    assert "category_distribution" in stats


def test_chatbot_tools_search():
    results = ComplaintChatbotTools.search_complaints(limit=3)
    assert len(results) <= 3
    if results:
        assert "complaint_id" in results[0]


def test_chatbot_bot_responses():
    bot = ComplaintIntelligenceBot()

    # Overview intent
    res_overview = bot.respond("show overview metrics")
    assert "Overview" in res_overview or "Complaints" in res_overview

    # Live Analysis intent
    res_analysis = bot.respond("Analyze: Unauthorized wire transfer drained $3,000 from my checking account!")
    assert "Report" in res_analysis or "Category" in res_analysis
    assert "Drafted Customer Response" in res_analysis
