from bs4 import BeautifulSoup
import base64

sources = ['Full Emoji List, v14.0.html', 'Full Emoji Modifier Sequences, v14.0.html']

for src in sources:
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
        soup = BeautifulSoup(content, 'html.parser')

    table = soup.find_all('tbody')[0]
    rows = table.find_all('tr')

    for row in rows:
        items = row.find_all('td')

        if len(items) == 15:
            try:
                code = items[1].a['name']  # unicode code
                raw = str(items[3].img['src'])
                image_binary = base64.b64decode(raw[22:])

                with open(f'imgs/{code}.png', 'wb') as img:
                    img.write(image_binary)
            except Exception as e:
                print(e)
                print(row)
