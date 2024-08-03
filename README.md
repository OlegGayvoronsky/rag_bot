<h1>Проект по созданию раг бота, который отвечает на вопросы пользователей</h1>
<br>Реализация бота состоит из двух главных этапов. На первом шаге на основе вопроса, предоставленного пользователем, с использованием <b>конвейера извлечения и ранжирования(retrieve and re-rank pipeline)</b>, как показано на рисунке ниже, находится наиболее соответствующий ему документ из базы знаний. На втором шаге <b>модель-генератор</b> дает ответ, используя вопрос пользователя и найденный документ.
<img src="https://github.com/OlegGayvoronsky/rag_bot/blob/main/images/photo_1.jpg">
В качестве модели bi-encoder используется <a href="https://huggingface.co/cointegrated/rubert-tiny2">cointegrated/rubert-tiny2</a>
<br>В качестве модели cross-encoder используется <a href="https://huggingface.co/DiTy/cross-encoder-russian-msmarco">DiTy/cross-encoder-russian-msmarco</a>
<br>В качестве генеративной модели используется <a href="https://huggingface.co/IlyaGusev/saiga_llama3_8b">IlyaGusev/saiga_llama3_8b</a>
<br>Чат для работы с ботом был написан с использованием фреймворка <a href="https://www.gradio.app/guides/quickstart">gradio</a>
<br>Весь проект развертывается при помощи докера.
<br>Результат работы проекта:
<img src="https://github.com/OlegGayvoronsky/rag_bot/blob/main/images/photo_2.jpg">
