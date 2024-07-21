240721

.env에

### app.py : 그냥 되는 버전

- 업로드하면 간단하게 q/a 할 수 있는 기능
- 그런데 너무 심한 추상화가 포함되었기 때문에 이해는 안될 수 있음 (load_qa_chain)

### app-custom-ragchain : 추상화를 조금 덜어낸 업그레이드 버전

- load_qa_chain 사용안했음

### runnablepassthrough-assign.py

- : input 자체도 chain의 결과로 보내주기 위해서는 .assign을 해줘서 dictionary를 관리하는 형태로 해줘야 함
- rag chain에서는 결국 질문, 컨텍스트, 정답 이 3가지 모두가 chain의 결과물로 호출되어야 하기 때문에 runnablepassthrough.assign이 많이 활용됨 (아니면 load_qa_chain 처럼 이미 그 로직이 내장되어 있거나)
