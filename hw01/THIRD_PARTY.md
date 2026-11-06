# Источники и права

Код PoetryGPT и исследовательские рецепты разработаны в этом проекте.
Лицензия собственного кода не назначена.

Корпуса RSD, PoetryCorpus, ru-poetry-dataset, corpus_1835, rus_elegies,
страницы Lib.ru/Классики, Викитеки и РВБ имеют разные происхождение и условия.
Общественное достояние произведения не означает одинаковых условий для
современной разметки, издания или базы. Конкретные URL/revision и сведения
о корпусах сохраняются в `reproduction/corpus/manifest.json` и метаданных, см. [данные](docs/dataset.md).
Эти сведения не заменяют проверку условий перед публичной раздачей полных текстов.

Фонетические ресурсы:

- RussianPoetryScansionTool, ревизия `9e577f036d970ba648f19d9d0c286547d9327902`, MIT;
- rusyllab, ревизия `c5d0ea405e9cc8233bd65503c26c0d2ad8b5cb57`;
- модели и native-зависимости перечислены с SHA в `artifacts/evaluators/rpst/version.json`;
- исходные LICENSE сторонних компонентов сохраняются вместе с ресурсами.

Снимок исходников RPST содержит Git LFS-указатели. Рабочие фонетические модели
расположены отдельно в `artifacts/evaluators/rpst/models/` и включены в архив инференса.
На Unix native-зависимости устанавливаются по `requirements-phonetics.txt`;
исходники и модели RPST остаются теми же.

Образцы оформления сверялись с [GitHub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories).
Чужой [mmot-task1](https://github.com/s02250557-study/mmot-task1/tree/33c1ffbff6adf326e79ec63e5dfb239a68085c8d)
использовался как сравнение подходов; чужой код и веса в PoetryGPT не копировались.
Организация команд и конфигов также сверялась с [nanoGPT](https://github.com/karpathy/nanoGPT)
и [LitGPT](https://github.com/Lightning-AI/litgpt). Их код в комплект не переносился.
