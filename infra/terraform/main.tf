# AWS for the search stack: model artifacts in S3, the listings index in OpenSearch, the service image in ECR.
# `terraform validate` runs in CI; nothing here is applied from CI.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "name" {
  type    = string
  default = "preloved-search"
}

variable "opensearch_instance" {
  type    = string
  default = "r6g.large.search"
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.name}-models"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled" # every published model stays retrievable for rollback
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "service" {
  name                 = "${var.name}-service"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_opensearch_domain" "listings" {
  domain_name    = var.name
  engine_version = "OpenSearch_2.13"

  cluster_config {
    instance_type          = var.opensearch_instance
    instance_count         = 3
    zone_awareness_enabled = true
    zone_awareness_config {
      availability_zone_count = 3
    }
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 100
    volume_type = "gp3"
  }

  encrypt_at_rest {
    enabled = true
  }

  node_to_node_encryption {
    enabled = true
  }

  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }
}

output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.listings.endpoint
}

output "ecr_repository" {
  value = aws_ecr_repository.service.repository_url
}
