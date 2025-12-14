{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:


   {% block methods %}


   .. rubric:: {{ _('Methods') }}


   .. autosummary::
      :toctree:
      :nosignatures:

   {% for item in methods %}
   {% if not (item == "__init__" or item in inherited_members or item == "cross_entropy" or item == "kl_divergence") %}
      ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}

   {% endblock %}

   {% block attributes %}


   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :template: autosummary/attribute.rst
      :toctree:

      {% for item in attributes %}
      {% if not item in inherited_members %}
         ~{{ name }}.{{ item }}
      {% endif %}
      {%- endfor %}


   {% endblock %}
