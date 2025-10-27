# Player Churn MLOps
[![CI/CD Pipeline](https://github.com/hzabun/player-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/hzabun/player-churn-mlops/actions/workflows/ci.yml)

> [!WARNING] Project is still WIP, lots of changes are expected

A production-ready MLOps project for predicting player churn in online games (Blade and Soul) using Prefect orchestration and Feast feature store.


## 🚀 Features

- **Data Pipeline**: Automated preprocessing of game logs to player-level features
- **Feature Store**: Feast-based feature management for online/offline serving
- **Orchestration**: Prefect flows for scheduling and monitoring
- **CI/CD**: GitHub Actions for testing, linting, and deployment
- **Testing**: Comprehensive unit tests with >90% coverage

## 📊 Pipeline

```
Raw Logs → Preprocess → Feature Store → Train → Predict → Deploy
```

## 📈 Roadmap

- [x] Data preprocessing pipeline
- [x] Feast feature store setup
- [x] Comprehensive unit tests
- [x] CI/CD with GitHub Actions
- [ ] Model training with LightGBM
- [ ] Containerizing pipeline
- [ ] Migration to AWS (S3, EKS, ECR)
- [ ] Model deployment on AWS EKS
- [ ] Feature drift monitoring
