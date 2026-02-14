# import click
# from core.logger import logger
# from data.spider_v1 import DatasetDownloader
# from sql_generator.sql_generator import SQLGenerator
# from schema_linker.schema_linker import SchemaLinker
#
#
#
# @click.group()
# def cli():
#     """Text-to-SQL System CLI"""
#     pass
#
#
# @cli.command()
# @click.option('--path', default='./data/spider', help='Dataset download path')
# @click.option('--force', is_flag=True, help='Force re-download')
# def download_spider(path, force):
#     """Download Spider dataset"""
#     downloader = DatasetDownloader(download_path=path)
#     result = downloader.download_from_kaggle(force_download=force)
#     logger.info(f"Dataset ready at: {result}")
#
#
# @cli.command()
# @click.argument('question')
# @click.option('--db-id', required=True, help='Database ID')
# @click.option('--model', default='codellama:7b', help='LLM model name')
# def query(question, db_id, model):
#     """Generate SQL from natural language question"""
#     # Initialize components
#     sql_generator = SQLGenerator(model=model)
#     schema_linker = SchemaLinker()
#
#     # Load schema for the database
#     schema = schema_linker.get_schema(db_id)
#
#     # Generate SQL
#     result = sql_generator.generate_sql(
#         question=question,
#         schema=schema,
#         db_id=db_id
#     )
#
#     click.echo(f"Generated SQL: {result.sql}")
#     click.echo(f"Confidence: {result.confidence}")
#
#
# @cli.command()
# @click.option('--split', default='dev', help='Dataset split to evaluate')
# def evaluate(split):
#     """Evaluate system on Spider dataset"""
#     from evaluation.evaluator import evaluate_on_spider
#     results = evaluate_on_spider(split=split)
#     click.echo(f"Execution accuracy: {results['exec_acc']:.2%}")
#
#
# if __name__ == "__main__":
#     cli()