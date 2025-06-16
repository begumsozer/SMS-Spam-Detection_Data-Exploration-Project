SMS Spam Detection - Prompt Engineering Project
This project is part of the MIT No Code AI Certificate Program that I participated in. The objective of the project is to analyze SMS messages and detect whether a given message is spam or not. This analysis was performed using a business interpretation approach combined with machine learning techniques, particularly classification with decision systems.

Project Overview
The dataset used contains over 5,500 text messages labeled as either "ham" (not spam) or "spam." Our goal is to build a prompt-engineered model that can accurately detect spam messages based on the text content.

We focus on:
Spam Rates: Understanding the proportion of spam in the dataset.

Message Patterns: Identifying keywords and structures common in spam.

Feature Engineering: Using a Bag-of-Words model for textual feature extraction.

A Decision Tree classifier is employed to generate interpretable results. Business insights derived from this model can be applied to email/SMS filtering tools, enhancing digital communication security.

Features
Data Exploration:
Class distribution visualization.

Frequency and percentage of spam vs ham messages.

Text Preprocessing:
Removal of English stopwords.

Transformation of message content using Bag-of-Words vectorization.

Predictive Modeling:
Train/test split for evaluating generalization.

Decision Tree Classifier with pruning to reduce overfitting.

Evaluation using accuracy, confusion matrix, and precision/recall scores.

Business Insights:
Identification of common spam traits.

Practical applications for telecom providers or digital marketing filters.

