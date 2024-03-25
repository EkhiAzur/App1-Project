from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PoolingClassificationModel(torch.nn.Module):
    
    def __init__(self, model_name, config = None, pooling_layer=-1, pooling_strategy="cls", torch_dtype = None, **kwargs):
        super(PoolingClassificationModel, self).__init__()

        # Load the base model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.config = self.model.config

        # Define the classification head
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 2)

        # CE loss will be used
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.pooling_layer = pooling_layer
        self.pooling_strategy = pooling_strategy

        # Choose the pooling strategy. Default is to choose CLS last embedding
        if self.pooling_strategy == "mean":
            self.pooling = lambda hidden_state: torch.sum(hidden_state,dim = 1) / torch.where(hidden_state != 0, 1,0).sum(dim = 1)

        elif self.pooling_strategy == "max":
            self.pooling = lambda hidden_state: torch.max(hidden_state, dim=1).values

        elif self.pooling_strategy == "min":
            self.pooling = lambda hidden_state: torch.min(hidden_state, dim=1).values

        elif self.pooling_strategy == "sum":
            self.pooling = lambda hidden_state: torch.sum(hidden_state, dim=1)

        elif self.pooling_strategy == "attention":
            self.attention_vector = torch.nn.Linear(self.model.config.hidden_size, 1)        
            self.pooling = self.attention_pooling

        elif self.pooling_strategy == "cls":
            self.pooling = lambda hidden_state: hidden_state[:,0,:]

        else:
            self.pooling = lambda hidden_state: hidden_state[:,0,:]
            logging.warning("Pooling strategy not recognized, using CLS token")


    def forward(self, input_ids, attention_mask, labels, **kwargs):

        # Get LM output
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Select the pooling layers values (default the last one)
        hidden_states = outputs.hidden_states[self.pooling_layer]
        
        # Apply the pooling function
        pooled_hidden_states = self.pooling(hidden_states)

        # Pass the output vector to the classification head
        pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier(pooled_hidden_states)

        if labels is not None:

            loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=pooled_hidden_states,
            )
        
        else:
            
            return SequenceClassifierOutput(
                logits=logits,
                hidden_states=pooled_hidden_states,
            )
        
    def attention_pooling(self, hidden_state):
        """
        Compute the attention based pooling
        
        Args:

            hidden_state (Tensor): The layer to apply the pooling

        Returns:

            pooled_hidden_states (Tensor): Vector calculated using attention based pooling strategy

        """
        # Calculate the weight of each last embedding vector
        attention_weights = torch.nn.functional.softmax(self.attention_vector(hidden_state), dim=1)
        # Calculate the weighted average based on the previous weights
        pooled_hidden_states = torch.sum(hidden_state * attention_weights, dim=1)
        return pooled_hidden_states

def load_model(runArgs, trainingArgs):
    """
    Load model and tokenizer

    Args:

        runArgs (RunArgs): RunArgs object containing the arguments to run the model
        trainingArgs (TrainingArguments): TrainingArguments object containing the arguments to train the model

    Returns:

        model (AutoModelForSequenceClassification): Model to train
        tokenizer (AutoTokenizer): Tokenizer to use

    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        runArgs.model_name,
        use_fast=True,
        add_prefix_space=True
    )

    model = PoolingClassificationModel( runArgs.model_name, runArgs.pooling_strategy="cls")

    return model, tokenizer
