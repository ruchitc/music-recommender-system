# Generated by Django 4.1.6 on 2023-03-06 13:51

import apps.endpoints.models
from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('endpoints', '0002_remove_mlalgorithm_created_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='endpoint',
            name='updated_at',
            field=apps.endpoints.models.AutoDateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='mlalgorithm',
            name='created_at',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='mlalgorithm',
            name='updated_at',
            field=apps.endpoints.models.AutoDateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='mlalgorithmstatus',
            name='updated_at',
            field=apps.endpoints.models.AutoDateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='mlrequest',
            name='updated_at',
            field=apps.endpoints.models.AutoDateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='endpoint',
            name='created_at',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='mlalgorithmstatus',
            name='created_at',
            field=models.DateField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='mlrequest',
            name='created_at',
            field=models.DateField(default=django.utils.timezone.now),
        ),
    ]
