## 가상환경

python3 -m venv venv
source venv/bin/activate

### app.py : 그냥 되는 버전

- 업로드하면 간단하게 q/a 할 수 있는 기능
- 그런데 너무 심한 추상화가 포함되었기 때문에 이해는 안될 수 있음 (load_qa_chain)

### app-custom-ragchain : 추상화를 조금 덜어낸 업그레이드 버전

- load_qa_chain 사용안했음
- 그런데 ragchain에 answer, context를 하나로 만들어버리면 streaming이 어려움
- 그래서 ragchain에서는 일단 결과를 answer만 나오게 하고, 나머지를 별도로 저장해야 streaming은 쉬워짐

### custom-ragchain-stream : 너무 assign을 이용해서 chain만 쓰려고 하다보니까 stream 기능이 없음

- 그래서 기능의 분화 (main chain에서는 answer 만 나오도록 함)를 통해 stream 달성

### custom-ragchain-stream-meta :

- pdf parsing logic을 잘 수정해서 (아니면 일반적인 pymupdf의 로직을 이용해서) metadata까지 포함되도록 함

### runnablepassthrough-assign.py

- : input 자체도 chain의 결과로 보내주기 위해서는 .assign을 해줘서 dictionary를 관리하는 형태로 해줘야 함
- rag chain에서는 결국 질문, 컨텍스트, 정답 이 3가지 모두가 chain의 결과물로 호출되어야 하기 때문에 runnablepassthrough.assign이 많이 활용됨 (아니면 load_qa_chain 처럼 이미 그 로직이 내장되어 있거나)

### summarize-pdf-nonstream.py

- 가장 간단한 형태의 요약 앱, 심지어 stream도 안됨

### summarize-pdf-stream.py

- 한단계 더 나아간 요약앱, langchain의 구조를 활용해서 stream이 됨
- invoke 대신 stream, write 대신 write_stream을 이용하면 됨

### summarize-pdf-withoptions.py

- 여러 옵션 중에서 선택 가능

pip freeze > requirements.txt

240725
발전된 형태가 custom-ragchain-stream-meta

- 여기서 한단계 더 발전한 형태가 rag-with-pdf.py
