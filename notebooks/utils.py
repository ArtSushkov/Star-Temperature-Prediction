# === Импорт библиотек ===

# Стандартные библиотеки Python
import warnings

# Основные библиотеки для анализа данных
import numpy as np
import pandas as pd

# Библиотеки для статической визуализации
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Библиотеки для интерактивной визуализации
import plotly.express as px
import plotly.graph_objects as go

# Библиотеки для работы в Jupyter
from IPython.display import HTML, display

# Библиотеки для статистического анализа
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Библиотеки для анализа корреляций
from phik.report import plot_correlation_matrix

# Настройка системы предупреждений
warnings.filterwarnings("ignore", category=FutureWarning)


# === Собственные функции и класс ===

def extended_describe(df, name="DataFrame"):
    """
    Возвращает расширенное описание датафрейма в виде HTML-таблицы.

    Добавляет полезные метрики к стандартному описанию:
    - mismatch%: отклонение среднего (mean) от медианы (50%) в процентах
    - rel_std%: относительное стандартное отклонение в процентах (std / mean * 100)
    - cat_top_ratio%: доля самого частого значения в категориальных столбцах (freq / count * 100)

    Параметры:
    ----------
    df : pd.DataFrame
        Входной датафрейм для анализа.
    name : str, optional (default="DataFrame")
        Название датафрейма, отображается в заголовке вывода.

    Возвращает:
    -----------
    None
        Результат выводится в виде HTML-таблицы в интерфейсе (например, Jupyter Notebook).

    Пример использования:
    ---------------------
    >>> extended_describe(messages_df, name="messages_df")

    Числовое описание данных: messages_df
    +--------------+--------+--------+----------+-------------+----------------+----------------+
    |              | count  | unique |   mean   |     std     |   mismatch%    |   rel_std%     |
    +--------------+--------+--------+----------+-------------+----------------+----------------+
    | column_1     | 100.00 | NaN    |  50.23   |   10.45     |     0.46       |     20.81      |
    | category_col | 100.00 | 5.00   |   NaN    |    NaN      |     NaN        |     NaN        |
    +--------------+--------+--------+----------+-------------+----------------+----------------+

    Примечание:
    ------------
    Функция работает как с числовыми, так и с категориальными столбцами.
    Требует библиотеки pandas и IPython для отображения HTML.
    """

    # Получаем стандартное описание
    desc = df.describe(include='all')

    # Создаем копию, чтобы не модифицировать исходный describe
    pivot = desc.copy()

    # Добавляем mismatch% для числовых столбцов
    if 'mean' in pivot.index and '50%' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            mismatch = ((pivot.loc['mean'] - pivot.loc['50%']) / pivot.loc['50%']) * 100
        mismatch.name = 'mismatch%'
        pivot = pd.concat([pivot, mismatch.to_frame().T])

    # Добавляем rel_std% для числовых столбцов
    if 'std' in pivot.index and 'mean' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            rel_std = (pivot.loc['std'] / pivot.loc['mean']) * 100
        rel_std.name = 'rel_std%'
        pivot = pd.concat([pivot, rel_std.to_frame().T])

    # Добавляем cat_top_ratio% для категориальных столбцов
    if 'freq' in pivot.index and 'count' in pivot.index:
        with pd.option_context('mode.use_inf_as_na', True):
            cat_ratio = (pivot.loc['freq'] / pivot.loc['count']) * 100
        cat_ratio.name = 'cat_top_ratio%'
        pivot = pd.concat([pivot, cat_ratio.to_frame().T])

    # Округляем и транспонируем
    styled_table = pivot.round(2).T

    # Выводим в HTML
    print(f'\033[1mЧисловое описание данных: {name}\033[0m')
    display(HTML(styled_table.to_html(index=True)))


def plot_distribution_with_boxplot(df, features, target_col, category_order=None, bins=None,
                                 auto_bins=False, log_scale=False, minor_category_threshold=0.05,
                                 show_category_types=True):
    """Визуализирует распределения признаков с разделением по категориям целевой переменной.

    Автоматически преобразует числовые целевые переменные в категориальные,
    если они содержат ≤10 уникальных значений. Это преобразование не влияет на исходный датафрейм.

    Создает сетку графиков:
    - Для каждого признака отображаются 2 строки:
      1) Гистограмма с наложениями категорий (stacked) и KDE-кривой (с возможностью
         использования второй оси Y для редких категорий, если они есть)
      2) Горизонтальный boxplot, где категории размещены по оси y
    - Цвета категорий согласованы между графиками и указаны в легенде
    - Для дисбалансированных данных редкие категории (ниже порога) отображаются на отдельной оси Y
    - Если малых категорий нет, вторая ось Y не создается, а подпись "Основная" в легенде не добавляется
    - Графики автоматически размещаются в сетке (до 4 столбцов)
    - Убираются пустые оси для незаполненных ячеек сетки
    - Настройки: сетка, поворот меток, оптимизация макета

    Args:
        df (pd.DataFrame): DataFrame с данными
        features (List[Tuple[str, str]]): Список кортежей (колонка, человекочитаемая метка)
        target_col (str): Название целевой колонки. Если тип числовой и содержит ≤10 уникальных значений,
                          будет преобразована в категориальную внутри функции
        category_order (Optional[List[str]]): Список категорий в нужном порядке. Если None,
            используется отсортированный порядок (по умолчанию None)
        bins (Optional[int]): Количество корзин для гистограмм. Если None и auto_bins=False,
            используется значение по умолчанию в sns.histplot (по умолчанию None)
        auto_bins (bool): Если True, для каждого признака количество корзин будет определяться
            автоматически с помощью правила Фридмана-Дьякониса (по умолчанию False)
        log_scale (bool or str or list): Если True, логарифмический масштаб применяется ко всем признакам.
            Если 'auto', применяется только к признакам с большим разбросом значений.
            Если список, указывает к каким именно признакам применить (по именам колонок).
            (по умолчанию False)
        minor_category_threshold (float): Порог для определения редких категорий (доля от общего числа записей).
            Категории с долей меньше этого значения будут отображаться на отдельной оси Y.
            (по умолчанию 0.05)
        show_category_types (bool): Если True, в легенде будет указано, является ли категория
            основной или редкой. **Если малых категорий нет, пометка "Основная" не добавляется.**
            (по умолчанию True)

    Notes:
        - features ожидает кортежи (колонка, человекочитаемая метка)
        - Для числовых целевых переменных с ≤10 уникальными значениями автоматически создаётся
          категориальная переменная (без изменения исходного DataFrame)
        - Если нет малых категорий, правая ось Y не создается, а легенда не содержит избыточных меток

    Examples:
        >>> # Пример 1: Базовое использование с автоматическим определением категорий
        >>> features = [('age', 'Возраст'), ('income', 'Доход')]
        >>> plot_distribution_with_boxplot(df, features, 'gender')

        >>> # Пример 2: Логарифмический масштаб для всех признаков
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale=True)

        >>> # Пример 3: Логарифмический масштаб только для определенных признаков
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale=['income'])

        >>> # Пример 4: Автоматическое определение признаков для логарифмического масштаба
        >>> plot_distribution_with_boxplot(df, features, 'gender', log_scale='auto')

        >>> # Пример 5: Настройка отображения редких категорий (менее 10% записей)
        >>> plot_distribution_with_boxplot(df, features, 'rare_category', minor_category_threshold=0.1)

        >>> # Пример 6: Отключение пометок категорий в легенде
        >>> plot_distribution_with_boxplot(df, features, 'category', show_category_types=False)

        >>> # Пример 7: Комбинированное использование параметров
        >>> plot_distribution_with_boxplot(
        ...     df,
        ...     features=[('age', 'Возраст'), ('income', 'Доход')],
        ...     target_col='membership_type',
        ...     minor_category_threshold=0.1,
        ...     log_scale='auto',
        ...     auto_bins=True
        ... )
    """

    # Проверка и преобразование целевой переменной
    target_series = df[target_col]
    unique_count = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    # Создаем временную целевую переменную
    if is_numeric and unique_count <= 10:
        transformed_target = target_series.astype(str)
        print(f"Целевой признак '{target_col}' преобразован в категориальный. "
              f"Уникальные значения: {sorted(transformed_target.unique())}")
    else:
        transformed_target = target_series.copy()

    # Получаем категории в нужном порядке (удалив строки с пропусками, если они имеются)
    if category_order is None:
        if is_numeric and unique_count <= 10:
            numeric_categories = sorted(df[target_col].dropna().unique())
            categories = [str(val) for val in numeric_categories]
        else:
            categories = sorted(transformed_target.dropna().unique())
    else:
        categories = category_order

    # Определяем основные и второстепенные категории
    category_counts = transformed_target.value_counts(normalize=True)
    major_categories = category_counts[category_counts >= minor_category_threshold].index.tolist()
    minor_categories = category_counts[category_counts < minor_category_threshold].index.tolist()

    # Подготовка цветовой палитры
    palette = plt.cm.Paired(np.linspace(0, 1, len(categories)))
    category_to_color = {cat: color for cat, color in zip(categories, palette)}

    # Формируем элементы легенды с указанием типа категории (если нужно и если есть малые категории)
    has_minor_categories = len(minor_categories) > 0
    should_show_types = show_category_types and has_minor_categories

    if should_show_types:
        legend_elements = [
            Patch(facecolor=color,
                 label=f"{cat} (Основная)" if cat in major_categories else f"{cat} (Малая)",
                 alpha=0.6)
            for cat, color in category_to_color.items()
        ]
    else:
        legend_elements = [
            Patch(facecolor=color, label=str(cat), alpha=0.6)
            for cat, color in category_to_color.items()
        ]

    # Настройка сетки графиков
    n_features = len(features)
    ncols = min(4, n_features)
    rows_per_feature = 2
    nrows = (n_features + ncols - 1) // ncols * rows_per_feature

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 3 * nrows),
        squeeze=False
    )

    # Функция для вычисления оптимального числа корзин по правилу Фридмана-Дьякониса
    def calculate_fd_bins(data):
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / (len(data) ** (1/3))
        if bin_width == 0:
            return 30
        return int(np.ceil((data.max() - data.min()) / bin_width))

    # Определяем, к каким признакам применять логарифмический масштаб
    if log_scale == 'auto':
        log_features = []
        for feature_col, _ in features:
            if pd.api.types.is_numeric_dtype(df[feature_col]):
                data = df[feature_col].dropna()
                if len(data) > 0 and data.min() > 0 and (data.max() / data.min() > 100):
                    log_features.append(feature_col)
        if log_features:
            print(f"Автоматически применен логарифмический масштаб к признакам: {log_features}")
    elif log_scale is True:
        log_features = [feature_col for feature_col, _ in features
                       if pd.api.types.is_numeric_dtype(df[feature_col])]
    elif isinstance(log_scale, list):
        log_features = log_scale
    else:
        log_features = []

    # Цикл по признакам для создания графиков
    for i, (feature_col, feature_label) in enumerate(features):
        col = i % ncols
        row_base = (i // ncols) * rows_per_feature

        # Определение количества корзин для текущего признака
        current_bins = bins
        if auto_bins and pd.api.types.is_numeric_dtype(df[feature_col]):
            data_clean = df[feature_col].dropna()
            if len(data_clean) > 0:
                current_bins = calculate_fd_bins(data_clean)
                print(f"Для признака '{feature_col}' автоматически выбрано {current_bins} корзин")

        use_log_scale = feature_col in log_features

        # Гистограмма
        ax_hist = axes[row_base, col]
        ax_hist_right = None  # Инициализируем как None

        # Рисуем основные категории на левой оси
        if major_categories:
            sns.histplot(
                data=df[df[target_col].isin(major_categories)],
                x=feature_col,
                hue=transformed_target,
                kde=True,
                multiple='stack',
                palette=category_to_color,
                ax=ax_hist,
                legend=False,
                bins=current_bins,
                log_scale=use_log_scale
            )

        # Только если есть малые категории — создаём правую ось и рисуем на ней
        if minor_categories:
            ax_hist_right = ax_hist.twinx()

            sns.histplot(
                data=df[df[target_col].isin(minor_categories)],
                x=feature_col,
                hue=transformed_target,
                kde=True,
                multiple='stack',
                palette=category_to_color,
                ax=ax_hist_right,
                legend=False,
                bins=current_bins,
                log_scale=use_log_scale
            )

            # Настраиваем правую ось
            ax_hist_right.spines['right'].set_color('gray')
            ax_hist_right.tick_params(axis='y', colors='gray')
            ax_hist_right.yaxis.label.set_color('gray')
            ax_hist_right.set_ylabel(
                f'Малые категории (<{minor_category_threshold*100:.0f}%)',
                color='gray',
                fontsize=8
            )

        # Настройка левой оси (основной)
        ax_hist.set_title(f'Распределение {feature_label}', fontsize=10)
        ax_hist.set_xlabel('')

        if use_log_scale:
            ax_hist.set_xscale('log')
            ax_hist.set_title(f'Распределение {feature_label} (log scale)', fontsize=10)

        # Подпись оси Y: только если нет малых категорий или если это первый график
        if minor_categories:
            if i == 0:
                ax_hist.set_ylabel(
                    f'Основные категории (≥{minor_category_threshold*100:.0f}%)',
                    fontsize=9
                )
            else:
                ax_hist.set_ylabel('')
        else:
            # Если малых категорий нет, просто подписываем как "Частота"
            if i == 0:
                ax_hist.set_ylabel('Частота', fontsize=9)
            else:
                ax_hist.set_ylabel('')

        # Легенда только на первом графике
        if i == 0:
            ax_hist.legend(
                handles=legend_elements,
                title=target_col,
                fontsize=8,
                title_fontsize=8,
                loc='upper right'
            )

        ax_hist.grid(axis='y', alpha=0.3)
        ax_hist.tick_params(axis='x', labelrotation=0)

        # Boxplot
        ax_box = axes[row_base + 1, col]
        sns.boxplot(
            data=df,
            x=feature_col,
            y=transformed_target,
            order=categories,
            palette=category_to_color,
            orient='h',
            ax=ax_box,
            width=0.6
        )

        if use_log_scale:
            ax_box.set_xscale('log')

        # Настройка boxplot
        if i == 0:
            ax_box.tick_params(axis='y', labelrotation=45, labelsize=8)
        else:
            ax_box.set_yticklabels([])

        ax_box.set_xlabel(feature_label, fontsize=9)
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.tick_params(axis='x', labelrotation=0)
        ax_box.set_ylabel('')

    # Скрытие пустых осей
    for row in range(nrows):
        for col in range(ncols):
            current_idx = (row // rows_per_feature) * ncols + col
            if current_idx >= n_features:
                axes[row, col].set_visible(False)

    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.show()


def create_sunburst_plot(df, category_columns=None):
    """Создаёт диаграмму Sunburst для визуализации иерархического распределения категориальных данных в DataFrame.

    Диаграмма Sunburst отображает вложенность категорий: каждый уровень вложенности соответствует
    одному из категориальных столбцов, а размер сектора пропорционален доле записей в этой категории.

    Если список `category_columns` не указан, функция автоматически определяет категориальные столбцы
    как столбцы с типом `object` или `category`. Пропущенные значения заменяются на 'Unknown',
    чтобы избежать ошибок при группировке и построении диаграммы.

    Особенности визуализации:
    - Цвет сектора зависит от доли категории (в процентах от общего числа записей)
    - Используется цветовая шкала 'Viridis' с отображением цветовой легенды
    - Всплывающие подсказки (hover) содержат название категории, количество записей и процент
    - Диаграмма адаптирована под чистый фон (прозрачный), подходит для встраивания в отчёты
    - Автоматическое разрешение коллизий имён (если метки повторяются, добавляются пробелы)

    Args:
        df (pd.DataFrame): Исходный DataFrame, содержащий данные.
        category_columns (Optional[List[str]]): Список названий столбцов, которые следует использовать
            как уровни иерархии в диаграмме. Порядок столбцов определяет порядок вложенности.
            Если None, используются все столбцы с типом `object` или `category`, в порядке их появления.
            (по умолчанию None)

    Returns:
        plotly.graph_objects.Figure: Готовая диаграмма Sunburst, которую можно отобразить в Jupyter,
        сохранить или встроить в веб-приложение.

    Notes:
        - Пропущенные значения (NaN) заменяются на строку 'Unknown' для корректного отображения
        - Если одинаковые метки встречаются на одном уровне, к ним добавляются пробелы для уникальности
        - Порядок столбцов в `category_columns` критичен: первый столбец — корень, последующие — дочерние уровни
        - Диаграмма использует относительные доли (в процентах), а не абсолютные значения
        - Цветовая шкала отражает долю категории (в %), что позволяет быстро оценить значимость секторов

    Examples:
        >>> # Пример 1: Автоматическое определение категориальных столбцов
        >>> fig = create_sunburst_plot(df)
        >>> fig.show()

        >>> # Пример 2: Указание конкретных столбцов в заданной иерархии
        >>> category_order = ['region', 'department', 'team']
        >>> fig = create_sunburst_plot(df, category_columns=category_order)
        >>> fig.show()

        >>> # Пример 3: Визуализация с пропущенными значениями (автоматически обрабатываются)
        >>> # Допустим, в 'department' есть NaN — они отобразятся как 'Unknown'
        >>> fig = create_sunburst_plot(df, category_columns=['country', 'city'])
        >>> fig.write_html("sunburst.html")  # Сохранение в HTML

        >>> # Пример 4: Использование в конвейере анализа
        >>> from plotly.offline import plot
        >>> fig = create_sunburst_plot(df[df['year'] == 2023], category_columns=['category', 'subcategory'])
        >>> plot(fig, filename='sunburst_2023.html', auto_open=False)

        >>> # Пример 5: Просмотр структуры данных перед визуализацией
        >>> print("Категориальные столбцы:", df.select_dtypes(include=['object']).columns.tolist())
        >>> fig = create_sunburst_plot(df, category_columns=['level1', 'level2', 'level3'])
        >>> fig.update_layout(title="Иерархия категорий")  # Дополнительная настройка
        >>> fig.show()
    """
    # Создаем копию DataFrame
    df_copy = df.copy()

    # Заполняем пропущенные значения 'Unknown', чтобы избежать ошибок при построении диаграммы
    df_copy.fillna('Unknown', inplace=True)

    # Определяем категориальные столбцы
    if category_columns is None:
        category_columns = df_copy.select_dtypes(include=['object']).columns.tolist()

    # Списки для диаграммы Sunburst
    labels, parents, values, text = [], [], [], []
    label_map = {}

    for i, column in enumerate(category_columns):
        group_columns = category_columns[:i + 1]
        df_group = df_copy.groupby(group_columns).size().reset_index(name='count')

        total_count = df_group['count'].sum()

        for _, row in df_group.iterrows():
            current_label = row[column]
            parent_label = "" if i == 0 else label_map[tuple(row[category_columns[:i]])]

            # Добавляем пробел, если текущая метка уже существует
            while current_label in labels:
                current_label += " "

            # Добавляем значения
            labels.append(current_label)
            parents.append(parent_label)
            percentage = row['count'] / total_count * 100
            values.append(percentage)
            text.append(f'{column}: {row["count"]} ({percentage:.2f}%)')
            label_map[tuple(row[category_columns[:i+1]])] = current_label

    # Создаем диаграмму Sunburst с градиентом цвета для процентного содержания
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Проценты")
        ),
        branchvalues="total",
        text=text,
        hovertemplate='<b>%{label}</b><br>%{text}<extra></extra>',
    ))

    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        width=800,
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def phik_correlation_matrix(df, target_col=None, threshold=0.9, output_interval_cols=True, interval_cols=None, cell_size=1.1):
    """Строит матрицу корреляции Фи-К (включая целевую переменную) и возвращает корреляции с целевой.

    Args:
        df (pd.DataFrame): DataFrame с данными для анализа
        target_col (str): Название столбца с целевой переменной
        threshold (float): Порог для выделения значимых корреляций (0.9 по умолчанию)
        output_interval_cols (bool): Возвращать ли список числовых непрерывных столбцов
        interval_cols (list): Список числовых непрерывных столбцов (если None, будет определен автоматически)
        cell_size (float): Дюйм на ячейку

    Returns:
        tuple: (correlated_pairs, interval_cols, phi_k_with_target) где:
            - correlated_pairs: DataFrame с парами коррелирующих признаков
            - interval_cols: Список числовых непрерывных столбцов (если output_interval_cols=True)
            - phi_k_with_target: Series с корреляциями признаков с целевой переменной

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phik import phik_matrix
        >>>
        >>> # Создаем тестовые данные
        >>> data = {
        ...     'price': [100, 200, 150, 300],  # Целевая переменная
        ...     'mileage': [50, 100, 75, 120],
        ...     'brand': ['A', 'B', 'A', 'C'],
        ...     'engine': [1.6, 2.0, 1.8, 2.5]
        ... }
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Анализ корреляций с ручным заданием числовых столбцов
        >>> result = phik_correlation_matrix(df, target_col='price', threshold=0.3, interval_cols=['mileage', 'engine'])
        >>>
        >>> # Получаем результаты:
        >>> correlated_pairs = result[0]  # Пары коррелирующих признаков
        >>> interval_cols = result[1]     # Числовые непрерывные столбцы
        >>> phi_k_with_target = result[2] # Корреляции с ценой
        >>>
        >>> print("Корреляции с ценой:")
        >>> print(phi_k_with_target.sort_values(ascending=False))
    """

    # Определение числовых непрерывных столбцов (если не заданы вручную)
    if interval_cols is None:
        interval_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if (df[col].nunique() > 50) or ((df[col] % 1 != 0).any())
        ]

    # Расчет полной матрицы корреляции (включая целевую переменную)
    phik_matrix = df.phik_matrix(interval_cols=interval_cols).round(2)

    # Получение корреляций с целевой переменной
    phi_k_with_target = None
    if target_col is not None and target_col in phik_matrix.columns:
        phi_k_with_target = phik_matrix[target_col].copy()
        # Удаляем корреляцию целевой с собой (всегда 1.0)
        phi_k_with_target.drop(target_col, inplace=True, errors='ignore')

    # Динамическое определение размера фигуры для подстройки размера ячеек
    num_cols = len(phik_matrix.columns)
    num_rows = len(phik_matrix.index)
    cell_size = cell_size  # Дюймов на ячейку
    figsize = (num_cols * cell_size, num_rows * cell_size)

    # Визуализация матрицы
    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map="Greens",
        title=r"Матрица корреляции $\phi_K$",
        fontsize_factor=1,
        figsize=figsize
    )
    plt.tight_layout()
    plt.show()

    # Фильтрация значимых корреляций (исключая целевую из пар)
    close_to_one = phik_matrix[phik_matrix >= threshold]
    close_to_one = close_to_one.where(
        np.triu(np.ones(close_to_one.shape), k=1).astype(bool)
    )

    # Удаление строк/столбцов с целевой переменной для анализа пар признаков
    if target_col is not None:
        close_to_one.drop(target_col, axis=0, inplace=True, errors='ignore')
        close_to_one.drop(target_col, axis=1, inplace=True, errors='ignore')

    # Преобразование в длинный формат
    close_to_one_stacked = close_to_one.stack().reset_index()
    close_to_one_stacked.columns = ["признак_1", "признак_2", "корреляция"]
    close_to_one_stacked = close_to_one_stacked.dropna(subset=["корреляция"])

    # Классификация корреляций
    def classify_correlation(corr):
        if corr >= 0.9: return "Очень высокая"
        elif corr >= 0.7: return "Высокая"
        elif corr >= 0.5: return "Заметная"
        elif corr >= 0.3: return "Умеренная"
        elif corr >= 0.1: return "Слабая"
        return "-"

    close_to_one_stacked["класс_корреляции"] = close_to_one_stacked["корреляция"].apply(
        classify_correlation
    )
    close_to_one_sorted = close_to_one_stacked.sort_values(
        by="корреляция", ascending=False
    ).reset_index(drop=True)

    if len(close_to_one_sorted) == 0 and threshold >= 0.9:
        print("\033[1mМультиколлинеарность между парами входных признаков отсутствует\033[0m")

    # Формирование результата
    result = [close_to_one_sorted]
    if output_interval_cols:
        result.append(interval_cols)
    if target_col is not None:
        result.append(phi_k_with_target)
    elif output_interval_cols:
        result.append(None)

    return tuple(result)


def vif(X, font_size=12):
    """Строит столбчатую диаграмму с коэффициентами инфляции дисперсии (VIF) для всех входных признаков.

    Args:
        X (pd.DataFrame): DataFrame с входными признаками для анализа.
        font_size (int): Размер шрифта для текстовых элементов графика (по умолчанию 12).

    Notes:
        - Коэффициент инфляции дисперсии (VIF) показывает степень мультиколлинеарности между признаками.
        - График отображается напрямую через matplotlib.

    Example:
        Пример использования функции:

        >>> import pandas as pd
        >>> from statsmodels.stats.outliers_influence import variance_inflation_factor
        >>> import statsmodels.api as sm
        >>>
        >>> # Создаем тестовый датафрейм
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [2, 4, 6, 8, 10],  # Полностью коррелирует с feature1
        ...     'feature3': [3, 6, 9, 12, 15]   # Частично коррелирует
        ... })
        >>>
        >>> # Вызываем функцию для анализа VIF
        >>> vif(data)
        >>>
        >>> # В результате будет показан график с VIF для каждого признака
        >>> # (feature2 будет иметь очень высокий VIF из-за полной корреляции с feature1)
    """
    # Кодируем категориальные признаки
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    # Добавляем константу для корректного расчета VIF
    X_with_const = sm.add_constant(X_encoded)

    # Вычисляем VIF для всех признаков, кроме константы (индексы начинаются с 1)
    vif = [variance_inflation_factor(X_with_const.values, i)
           for i in range(1, X_with_const.shape[1])]  # Исключаем константу (0-й столбец)

    # Построение графика с использованием исходных названий признаков (без константы)
    num_features = X_encoded.shape[1]
    fig_width = num_features * 1.2
    fig_height = 12

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(x=X_encoded.columns, y=vif)

    # Настройки графика
    ax.set_ylabel('VIF', fontsize=font_size)
    ax.set_xlabel('Входные признаки', fontsize=font_size)
    plt.title('Коэффициент инфляции дисперсии для входных признаков (VIF)', fontsize=font_size)

    # Метки на осях
    plt.xticks(rotation=90, ha='right', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Добавляем значения на столбцы (опционально)
    # ax.bar_label(ax.containers[0], fmt='%.2f', padding=3, fontsize=font_size)

    plt.tight_layout()
    plt.show()