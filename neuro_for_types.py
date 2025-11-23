import os
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import matplotlib.pyplot as plt
import json

# Параметры
MAX_LENGTH = 100
HIDDEN_SIZE = 512
EMBEDDING_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
GRAD_CLIP = 1.0
DROPOUT_RATE = 0.2

# Загрузка списков типов
try:
    with open('type_main_list.json', 'r', encoding='utf-8') as f:
        type_main_list = json.load(f)
    with open('type_pod_list.json', 'r', encoding='utf-8') as f:
        type_pod_list = json.load(f)
    print("✅ Типы загружены успешно")
except FileNotFoundError:
    print("⚠️ Файлы типов не найдены, используем пустые списки")
    type_main_list = []
    type_pod_list = []

class ImprovedTextNormalizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4

        # Добавляем основные типы в словарь с приоритетом
        self._add_priority_words()

    def _add_priority_words(self):
        """Добавляем основные типы и подтипы в словарь"""
        priority_words = set()

        # Добавляем основные типы
        for main_type in type_main_list:
            if main_type:  # Проверяем, что строка не пустая
                priority_words.add(main_type.lower())
                # Разбиваем составные типы
                if '/' in main_type:
                    parts = main_type.split('/')
                    for part in parts:
                        if part.strip():
                            priority_words.add(part.strip().lower())
                if '-' in main_type:
                    parts = main_type.split('-')
                    for part in parts:
                        if part.strip():
                            priority_words.add(part.strip().lower())

        # Добавляем подтипы
        for pod_type in type_pod_list:
            if pod_type:  # Проверяем, что строка не пустая
                priority_words.add(pod_type.lower())
                if '-' in pod_type:
                    parts = pod_type.split('-')
                    for part in parts:
                        if part.strip():
                            priority_words.add(part.strip().lower())

        # Добавляем приоритетные слова в словарь
        for word in sorted(priority_words):
            if word and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def build_vocab(self, texts):
        """Строим словарь на основе текстов"""
        words = []
        for text in texts:
            # Улучшенная токенизация
            tokens = self._tokenize_text(text)
            words.extend(tokens)

        word_counts = Counter(words)

        # Добавляем слова в словарь (только те, которых еще нет)
        for word, count in word_counts.items():
            if word not in self.word2idx and count >= 1:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def _tokenize_text(self, text):
        """Улучшенная токенизация текста"""
        # Заменяем запятые и точки на пробелы
        text = re.sub(r'[.,]', ' ', text)
        # Разбиваем на слова, сохраняя дефисы
        tokens = re.findall(r'[а-яa-zё0-9-]+', text.lower())
        return tokens

    def text_to_sequence(self, text):
        """Преобразование текста в последовательность"""
        tokens = self._tokenize_text(text)
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        return [self.word2idx['<SOS>']] + sequence + [self.word2idx['<EOS>']]

    def sequence_to_text(self, sequence):
        """Преобразование последовательности в текст"""
        tokens = []
        for idx in sequence:
            if idx == self.word2idx['<SOS>']:
                continue
            if idx == self.word2idx['<EOS>']:
                break
            if idx == self.word2idx['<PAD>']:
                continue
            tokens.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(tokens)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 3, hidden_size)  # hidden + encoder_outputs (hidden*2)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size] - из декодера
        # encoder_outputs: [batch_size, seq_len, hidden_size * 2] - из энкодера

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat hidden state for every source word
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hidden_size]

        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        return torch.softmax(attention, dim=1)

class ImprovedEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_rate):
        super(ImprovedEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True,
                         bidirectional=True, num_layers=2, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(x))
        # embedded: [batch_size, seq_len, embedding_size]

        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, seq_len, hidden_size * 2]
        # hidden: [num_layers * num_directions, batch_size, hidden_size] = [4, batch_size, hidden_size]

        # Combine bidirectional hidden states from last layer
        # Берем последние два hidden states (forward и backward из последнего слоя)
        hidden_forward = hidden[-2]  # [batch_size, hidden_size] - forward из последнего слоя
        hidden_backward = hidden[-1]  # [batch_size, hidden_size] - backward из последнего слоя

        hidden_combined = torch.tanh(self.fc(
            torch.cat((hidden_forward, hidden_backward), dim=1)
        ))
        # hidden_combined: [batch_size, hidden_size]

        # Для декодера нужно вернуть hidden в формате [num_layers, batch_size, hidden_size]
        # Создаем начальный hidden state для декодера
        decoder_hidden = hidden_combined.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, decoder_hidden

class ImprovedDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_rate):
        super(ImprovedDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embedding_size + hidden_size * 2, hidden_size,
                         batch_first=True, num_layers=1, dropout=0.0)  # Убрали dropout для одного слоя
        self.fc_out = nn.Linear(hidden_size + hidden_size * 2 + embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, hidden, encoder_outputs):
        # x: [batch_size]
        # hidden: [1, batch_size, hidden_size] - от энкодера
        # encoder_outputs: [batch_size, seq_len, hidden_size * 2]

        x = x.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_size]

        # Calculate attention weights
        # hidden: [1, batch_size, hidden_size] -> берем [0] для получения [batch_size, hidden_size]
        attn_weights = self.attention(hidden[0], encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        # Calculate context vector
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size * 2]

        # Combine embedded input and context
        gru_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embedding_size + hidden_size * 2]

        # GRU forward pass
        output, hidden = self.gru(gru_input, hidden)
        # output: [batch_size, 1, hidden_size]
        # hidden: [1, batch_size, hidden_size]

        # Final prediction
        output_flat = output.squeeze(1)  # [batch_size, hidden_size]
        context_flat = context.squeeze(1)  # [batch_size, hidden_size * 2]
        embedded_flat = embedded.squeeze(1)  # [batch_size, embedding_size]

        combined = torch.cat((output_flat, context_flat, embedded_flat), dim=1)
        prediction = self.fc_out(combined)  # [batch_size, vocab_size]

        return prediction, hidden, attn_weights.squeeze(1)

class ImprovedSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(ImprovedSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)

        # Encoder forward
        encoder_outputs, hidden = self.encoder(source)

        # First input to decoder is <SOS> token
        x = target[:, 0]

        for t in range(1, target_len):
            output, hidden, _ = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t] = output

            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            x = target[:, t] if teacher_force else top1

        return outputs

# Датасет
class ImprovedNormalizationDataset(Dataset):
    def __init__(self, source_texts, target_texts, source_normalizer, target_normalizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_normalizer = source_normalizer
        self.target_normalizer = target_normalizer

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_seq = self.source_normalizer.text_to_sequence(self.source_texts[idx])
        target_seq = self.target_normalizer.text_to_sequence(self.target_texts[idx])

        source_padded = self.pad_sequence(source_seq, MAX_LENGTH)
        target_padded = self.pad_sequence(target_seq, MAX_LENGTH)

        return torch.tensor(source_padded, dtype=torch.long), torch.tensor(target_padded, dtype=torch.long)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return sequence + [0] * (max_length - len(sequence))
        else:
            return sequence[:max_length]

def order_prediction_by_rules(predicted_text):
    """Упорядочиваем предсказание по правилам: основной тип первый, затем подтипы"""
    words = predicted_text.split()

    if not words:
        return predicted_text

    # Ищем основной тип
    main_types_found = []
    pod_types_found = []
    other_words = []

    for word in words:
        word_lower = word.lower()
        # Проверяем, является ли слово основным типом
        is_main_type = any(main_type.lower() == word_lower for main_type in type_main_list)
        # Проверяем, является ли слово подтипом
        is_pod_type = any(pod_type.lower() == word_lower for pod_type in type_pod_list)

        if is_main_type:
            main_types_found.append(word)
        elif is_pod_type:
            pod_types_found.append(word)
        else:
            other_words.append(word)

    # Собираем результат: основной тип первый, затем подтипы, затем остальные слова
    result_parts = []

    # Добавляем основной тип (если нашли)
    if main_types_found:
        result_parts.append(main_types_found[0])  # Берем первый найденный основной тип
        # Остальные основные типы считаем подтипами
        pod_types_found.extend(main_types_found[1:])

    # Добавляем подтипы
    result_parts.extend(pod_types_found)

    # Добавляем остальные слова
    result_parts.extend(other_words)

    return '|'.join(result_parts)

# Функция обучения
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for i, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(source, target)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Улучшенная функция предсказания
def improved_predict(model, text, source_normalizer, target_normalizer, device):
    model.eval()

    with torch.no_grad():
        sequence = source_normalizer.text_to_sequence(text)
        if len(sequence) > MAX_LENGTH:
            sequence = sequence[:MAX_LENGTH]
        else:
            sequence = sequence + [0] * (MAX_LENGTH - len(sequence))

        source_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)

        encoder_outputs, hidden = model.encoder(source_tensor)

        outputs = [target_normalizer.word2idx['<SOS>']]

        for _ in range(MAX_LENGTH - 1):
            x = torch.tensor([outputs[-1]], dtype=torch.long).to(device)
            output, hidden, _ = model.decoder(x, hidden, encoder_outputs)

            predicted = output.argmax(1).item()
            outputs.append(predicted)

            if predicted == target_normalizer.word2idx['<EOS>']:
                break

        predicted_text = target_normalizer.sequence_to_text(outputs)

        # Пост-обработка: упорядочивание по правилам
        return order_prediction_by_rules(predicted_text)

# Безопасная загрузка модели
def safe_load_model(model_path, device):
    """Безопасная загрузка модели с обработкой custom классов"""
    try:
        # Сначала пробуем загрузить с weights_only=True
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        return checkpoint
    except Exception as e:
        print(f"⚠️ Weights-only loading failed: {e}")
        print("🔄 Trying with safe_globals context...")

        # Используем safe_globals для разрешения TextNormalizer
        with torch.serialization.safe_globals([ImprovedTextNormalizer]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        return checkpoint

# Основная функция обучения
def improved_main(resume_training=False, model_path='improved_type_fixer_model.pth', total_epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    if device.type == 'cuda':
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎯 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        torch.cuda.empty_cache()

    # Загрузка данных
    df = pd.read_excel('types_train.xlsx', sheet_name='river')
    source_texts = df['Было'].astype(str).tolist()
    target_texts = df['Стало'].astype(str).tolist()

    print(f"📊 Loaded {len(source_texts)} examples")

    # Создание улучшенных нормализаторов
    source_normalizer = ImprovedTextNormalizer()
    target_normalizer = ImprovedTextNormalizer()

    source_normalizer.build_vocab(source_texts)
    target_normalizer.build_vocab(target_texts)

    print(f"📚 Source vocabulary size: {source_normalizer.vocab_size}")
    print(f"📚 Target vocabulary size: {target_normalizer.vocab_size}")

    # Разделение на train/val
    train_source, val_source, train_target, val_target = train_test_split(
        source_texts, target_texts, test_size=0.1, random_state=42
    )

    print(f"🎯 Train size: {len(train_source)}, Val size: {len(val_source)}")

    # Создание датасетов
    train_dataset = ImprovedNormalizationDataset(train_source, train_target, source_normalizer, target_normalizer)
    val_dataset = ImprovedNormalizationDataset(val_source, val_target, source_normalizer, target_normalizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Создание модели
    encoder = ImprovedEncoder(source_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    decoder = ImprovedDecoder(target_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    model = ImprovedSeq2Seq(encoder, decoder, device).to(device)

    # Оптимизатор и планировщик
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    start_epoch = 0
    train_losses = []

    if resume_training:
        print(f"🔄 Resuming training from {model_path}")
        checkpoint = safe_load_model(model_path, device)

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_losses = checkpoint.get('train_losses', [])
            start_epoch = checkpoint.get('epoch', len(train_losses))
            print(f"📅 Continuing from epoch {start_epoch}")

    print(f"🧠 Improved model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🎯 Training for {total_epochs - start_epoch} additional epochs (total: {total_epochs})")

    # Функция потерь
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Обучение
    print("🚀 Starting training...")

    for epoch in range(start_epoch, total_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Обновление learning rate
        scheduler.step(train_loss)

        print(f'📈 Epoch: {epoch + 1:03}/{total_epochs}, Train Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Тестирование каждые 10 эпох
        if val_source and (epoch + 1) % 10 == 0:
            print(f"🔍 Test Results (Epoch {epoch + 1}):")

            test_indices = [
                epoch % len(val_source),
                (epoch + 10) % len(val_source),
                (epoch + 20) % len(val_source)
            ]

            for i, idx in enumerate(test_indices):
                test_text = val_source[idx]
                target_text = val_target[idx]

                prediction = improved_predict(model, test_text, source_normalizer, target_normalizer, device)
                print(f"   Example {i + 1}:")
                print(f"      Input: {test_text}")
                print(f"      Target: {target_text}")
                print(f"      Prediction: {prediction}")
                print('-----------------------------')

        # Сохранение модели каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'source_normalizer': source_normalizer,
                'target_normalizer': target_normalizer,
                'train_losses': train_losses,
                'epoch': epoch + 1
            }, f'improved_type_fixer_model_epoch_{epoch + 1}.pth')
            print(f"💾 Checkpoint saved: improved_type_fixer_model_epoch_{epoch + 1}.pth")

    # Финальное сохранение модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'source_normalizer': source_normalizer,
        'target_normalizer': target_normalizer,
        'train_losses': train_losses,
        'epoch': total_epochs
    }, 'improved_type_fixer_model.pth')

    print("✅ Training completed! Model saved.")

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('improved_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# Функция для предсказания
def improved_predict_text(input_text, model_path='improved_type_fixer_model.pth', device=None, pr=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if pr:
        print(f"🔧 Using device: {device}")

    try:
        # Загружаем чекпоинт модели
        checkpoint = safe_load_model(model_path, device)

        # Получаем нормализаторы из чекпоинта
        source_normalizer = checkpoint['source_normalizer']
        target_normalizer = checkpoint['target_normalizer']

        # Создаем модель
        encoder = ImprovedEncoder(source_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
        decoder = ImprovedDecoder(target_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
        model = ImprovedSeq2Seq(encoder, decoder, device).to(device)

        # Загружаем веса модели
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if pr:
            print("✅ Model loaded successfully!")
            print(f"📚 Source vocab size: {source_normalizer.vocab_size}")
            print(f"📚 Target vocab size: {target_normalizer.vocab_size}")

        # Выполняем предсказание
        result = improved_predict(model, input_text, source_normalizer, target_normalizer, device)

        if pr:
            print(f"\n🎯 Input: {input_text}")
            print(f"🎯 Output: {result}")

        return result

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model file exists and is compatible with this code version")
        return None

# Функция для пакетного предсказания
def improved_predict_list(input_list, model_path='improved_type_fixer_model.pth', device=None, pr=False):
    results = []

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if pr:
        print(f"🔧 Using device: {device}")

    try:
        # Загружаем чекпоинт модели
        checkpoint = safe_load_model(model_path, device)

        # Получаем нормализаторы из чекпоинта
        source_normalizer = checkpoint['source_normalizer']
        target_normalizer = checkpoint['target_normalizer']

        # Создаем модель
        encoder = ImprovedEncoder(source_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
        decoder = ImprovedDecoder(target_normalizer.vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
        model = ImprovedSeq2Seq(encoder, decoder, device).to(device)

        # Загружаем веса модели
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if pr:
            print("✅ Model loaded successfully!")
            print(f"📚 Source vocab size: {source_normalizer.vocab_size}")
            print(f"📚 Target vocab size: {target_normalizer.vocab_size}")

        # Выполняем предсказание для каждого элемента
        for i, text in enumerate(input_list):
            if i % 10 == 0 and pr:
                print(f"Processing {i}/{len(input_list)}")

            result = improved_predict(model, text, source_normalizer, target_normalizer, device)
            results.append(result)

            if pr:
                print(f"\n🎯 Input: {text}")
                print(f"🎯 Output: {result}")

        return results

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model file exists and is compatible with this code version")
        return None

# Класс для работы с результатами
class ImprovedTypesResult:
    def __init__(self, df, ti):
        self.df = df
        self.ti = ti

    def predicts(self, model_path='improved_type_fixer_model.pth'):
        input_list = self.df.iloc[:, 0].tolist()
        predict_results = improved_predict_list(input_list, model_path=model_path)
        self.df.iloc[:, 1] = predict_results

    def fixing(self):
        for row in range(len(self.df)):
            value0 = str(self.df.iloc[row, 1]).strip()
            # Уже обработано в order_prediction_by_rules
            self.df.iloc[row, 1] = value0

# Функция для продолжения обучения
def continue_improved_training(model_path='improved_type_fixer_model.pth', additional_epochs=50):
    """Продолжение обучения существующей модели"""
    improved_main(resume_training=True, model_path=model_path, total_epochs=additional_epochs)

# Тестирование
def test_improved_model(test_list):


    print("🧪 Testing improved model...")
    results = improved_predict_list(test_list)

    print("\n📋 Final Results:")
    for i, (input_text, output_text) in enumerate(zip(test_list, results)):
        print(f"{i+1}. Input: {input_text}")
        print(f"   Output: {output_text}")
        print()

#токенизация, проверка
def test_tokenizer(test_texts):
    """Тестируем токенизацию на примерах"""
    normalizer = ImprovedTextNormalizer()


    print("🔍 Тестирование токенизации:")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        tokens = normalizer._tokenize_text(text)
        sequence = normalizer.text_to_sequence(text)
        decoded = normalizer.sequence_to_text(sequence)

        print(f"{i}. Исходный текст: {text}")
        print(f"   Токены: {tokens}")
        print(f"   Последовательность: {sequence}")
        print(f"   Декодировано: {decoded}")
        print(f"   Длина: {len(tokens)} токенов")
        print("-" * 30)

def test_vocabulary(test_texts, test_word):
    """Проверяем построение словаря"""
    # Создаем нормализатор
    normalizer = ImprovedTextNormalizer()

    # Строим словарь
    normalizer.build_vocab(test_texts)

    print("📚 Словарь:")
    print(f"Размер словаря: {normalizer.vocab_size}")
    print("\nСлова в словаре:")
    for i, (word, idx) in enumerate(list(normalizer.word2idx.items())[:]):  # Покажем первые 20
        print(f"  {idx}: '{word}'")

    # Проверяем обратное преобразование
    print(f"\n🔁 Проверка обратного преобразования:")
    if test_word in normalizer.word2idx:
        idx = normalizer.word2idx[test_word]
        reconstructed = normalizer.idx2word[idx]
        print(f"  '{test_word}' -> {idx} -> '{reconstructed}'")

def test_special_tokens(test_list):
    """Проверяем работу специальных токенов"""
    normalizer = ImprovedTextNormalizer()
    for test_text in test_list:
    #test_text = "тестовый текст"
      sequence = normalizer.text_to_sequence(test_text)

      print("🎯 Проверка специальных токенов:")
      print(f"Исходный текст: '{test_text}'")
      print(f"Последовательность: {sequence}")
      print(f"Первый токен (должен быть <SOS>): {normalizer.idx2word.get(sequence[0], 'UNKNOWN')}")
      print(f"Последний токен (должен быть <EOS>): {normalizer.idx2word.get(sequence[-1], 'UNKNOWN')}")

      # Проверяем декодирование
      decoded = normalizer.sequence_to_text(sequence)
      print(f"Декодировано: '{decoded}'")

def comprehensive_tokenizer_test(test_list):
    """Комплексное тестирование токенизатора"""
    print("🧪 Комплексное тестирование токенизатора")
    print("=" * 50)

    # Запускаем все тесты
    test_special_tokens(test_list)
    print("\n")
    test_tokenizer(test_list)
    #print("\n")
    #test_vocabulary()

def debug_tokenization_issue():
    """Диагностика проблемы с токенизацией"""
    normalizer = ImprovedTextNormalizer()

    # Тестовый текст
    test_text = "Несамоходная сухогрузная маломерное баржа"

    print("🔍 Диагностика токенизации:")
    print("=" * 50)

    # 1. Токенизация
    tokens = normalizer._tokenize_text(test_text)
    print(f"1. Токены: {tokens}")

    # 2. Построение словаря
    #normalizer.build_vocab([test_text])
    print(f"2. Размер словаря: {normalizer.vocab_size}")

    # 3. Проверка каждого токена
    print(f"3. Проверка токенов в словаре:")
    for token in tokens:
        exists = token in normalizer.word2idx
        idx = normalizer.word2idx.get(token, normalizer.word2idx['<UNK>'])
        print(f"   '{token}' -> в словаре: {exists}, индекс: {idx}")

    # 4. Преобразование в последовательность
    sequence = normalizer.text_to_sequence(test_text)
    print(f"4. Последовательность: {sequence}")

    # 5. Декодирование обратно
    decoded = normalizer.sequence_to_text(sequence)
    print(f"5. Декодировано: '{decoded}'")

    # 6. Вывод части словаря
    print(f"6. Часть словаря (слова из текста):")
    for word, idx in normalizer.word2idx.items():
        if word in tokens or any(token in word for token in tokens):
            print(f"   {idx}: '{word}'")

#



if __name__ == "__main__":
    # Выберите один из вариантов:

    # 1. Обучить модель с нуля
    #improved_main()

    # 2. Продолжить обучение
    #continue_improved_training('improved_type_fixer_model.pth', additional_epochs=120)

    test_list = [
        "Несамоходная сухогрузная маломерное баржа"
    ]

    # 3. Протестировать модель
    #test_improved_model(test_list)

    # тестирование токенов
    #comprehensive_tokenizer_test(test_list)
    #test_vocabulary(test_list,"маломерное")
    #debug_tokenization_issue()

    # 4. Обработать Excel файл
    i = 0
    file_path = f"types_result_{i}.xlsx"
    file_path_new = f"types_result_{i+1}.xlsx"
    df = (pd.read_excel(file_path, sheet_name=0)).copy()
    tr = ImprovedTypesResult(df, i)
    tr.predicts()
    tr.fixing()
    with pd.ExcelWriter(file_path_new, engine="openpyxl") as writer:
      tr.df.to_excel(writer, sheet_name="result", index=False)