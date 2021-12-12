import koco
from hanspell import spell_checker


# path
base_path = '/content/drive/MyDrive/AIProject'
train_csv_path = base_path + '/train.csv'
valid_csv_path = base_path + '/valid.csv'
unlabeled_train_csv_path = base_path + '/unlabeled_train.csv'


# load data
train_dev = koco.load_dataset('korean-hate-speech', mode='train_dev')
train_set = train_dev['train']
valid_set = train_dev['dev']
unlabeled_set = koco.load_dataset('korean-hate-speech', mode='unlabeled')


# check spell
for i, item in enumerate(train_set):
    try:
        spell_checked = spell_checker.check(item['comments'])
        train_set[i]['comments'] = spell_checked.checked
    except Exception as e:
        print('train_set:', i) # 122, 6115 ParseError by &
        print(e)

for i, item in enumerate(valid_set):
    try:
        spell_checked = spell_checker.check(item['comments'])
        valid_set[i]['comments'] = spell_checked.checked
    except Exception as e:
        print('valid_set:', i)
        print(e)

for i, item in enumerate(unlabeled_set):
    try:
        spell_checked = spell_checker.check(item['comments'])
        valid_set[i]['comments'] = spell_checked.checked
    except Exception as e:
        print('valid_set:', i)
        print(e)

        
# save data
labels = ['comments', 'contain_gender_bias', 'bias', 'hate', 'news_title']

with open(train_csv_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=labels)
    writer.writeheader()
    for elem in train_set:
        writer.writerow(elem)

with open(valid_csv_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=labels)
    writer.writeheader()
    for elem in valid_set:
        writer.writerow(elem)

with open(valid_csv_path, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=labels)
    writer.writeheader()
    for elem in valid_set:
        writer.writerow(elem)