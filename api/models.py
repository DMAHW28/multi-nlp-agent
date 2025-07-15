import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,  max_len,d_model)

        pe[ 0,:, 0::2] = torch.sin(position * div_term)
        pe[ 0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x ):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # On ajoute l'encodage de positionel à l'entrée
        x = x + self.pe[:,:x.size(1)]
        # On applique la régularisation
        return self.dropout(x)

class TextClassifierTransformer(nn.Module):
    def __init__(self, vocab_size: int= 5000, d_model: int=64, dim_feedforward = 4 * 64,num_layers: int=2, output_dim: int=6, n_head: int=4, dropout: float=0.2, batch_first=True):
        super(TextClassifierTransformer, self).__init__()
        self.d_model = d_model
        # On définit l'embedding de  et l'encodeur positionnel
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # On définit la couche transformer encodeur
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward,batch_first=batch_first)
        # On définit le transformer encodeur avec num_encoder_layers couches comme celle définie
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # On définit la couche linéaire pour la prédiction
        self.fc = nn.Linear(d_model, output_dim)
        # On initialise les poids des modèles
        self._init_weights()

    def _init_weights(self):
        # On initialise les paramètres à partir la distribution xavier
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, source, attention_mask):
        # On convertit en vecteurs d'embedding
        source_embed = self.embedding(source) * math.sqrt(self.d_model)
        # On ajoute l'encodage positionnel
        source_positional = self.pos_encoder(source_embed)
        # On encode avec le masque des séquences
        z = self.transformer_encoder(src=source_positional, src_key_padding_mask=~attention_mask)
        # On projete pour la prediction des sentiments
        return self.fc(z[:, 0, :])