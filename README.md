# MuzNeboZena
Machine learning model nátrénovaný v tensorflow, který dokáže rozpoznat obličej ženy a muže. (Snad) dostupný na adrese http://muznebozena.wz.cz/
Jak byl vytvořen?
Nejdříve byl stažen dataset, ze kterého byl v Pythonu pomocí tensorflow.py a keras vytvořen fungující model. Ten byl natrénován na 7 epoch a exportován pomocí tensorflowjs knihovny do souboru model.json, který lze použít v prohlížeči. V prohlížeči uživatel zadá obrázky a přímo na jeho počítači je vypočítán výsledek. Nic se neposílá na webserver, prohlížeč načte model a použijeho na zpracování obrázků, které uživatel zadá.

Machine learning model for gender detection written in tensorflow. Should be up and working at http://muznebozena.wz.cz/
How did I make it?
First, I downloaded a dataset and used tensorflow to train a model. I trained it for 7 epochs and exported it into a model.json file using the tensorflowjs Python library. This way, I can use tensorflow.js (javascript library) to run it in the browser on the side of the client. There is no server-side machine learning, everything including gathering images is done at the client.
