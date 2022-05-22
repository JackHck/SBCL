# subclass-balancing-contrastive-learning
This repository provides the  code for paper: <br>
**Subclass-balancing Contrastive Learning for Long-tailed Recognition**
# Overview
In this paper, we prospose subclass-balancing contrastive learning (SBCL),
a novel supervised contrastive learning defined on subclasses, which are the clusters within each
head class, have comparable size as tail classes, and are adaptively updated during the training.
Instead of sacrificing instance-balance for class-balance, our method achieves both instance- and
subclass-balance by exploring the head-class structure in the learned representation space of the
model-in-training. In particular, we propose a bi-granularity contrastive loss that enforces a sample
(1) to be closer to samples from the same subclass than all the other samples; and (2) to be closer to
samples from a different subclass but the same class than samples from any other subclasses. While
the former learns representations with balanced and compact subclasses, the latter preserves the class
structure on subclass level by encouraging the same class’s subslasses to be closer to each other than
to any different class’s subclasses. Hence, it can learn an accurate classifier distinguishing original
classes while enjoy both the instance- and subclass-balance.
